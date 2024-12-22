use crate::FSRS;
use crate::error::{FSRSError, Result};
use crate::inference::{DECAY, FACTOR, ItemProgress, Parameters, S_MAX, S_MIN, next_interval};
use crate::model::check_and_fill_parameters;
use crate::parameter_clipper::clip_parameters;
use burn::tensor::backend::Backend;
use itertools::{Itertools, izip};
use ndarray::Array1;
use ndarray_rand::rand_distr::Distribution;
use priority_queue::PriorityQueue;
use rand::Rng;
use rand::{SeedableRng, distributions::WeightedIndex, rngs::StdRng};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::cmp::min;
use std::collections::HashMap;

trait Round {
    fn to_2_decimal(self) -> f32;
}

impl Round for f32 {
    fn to_2_decimal(self) -> f32 {
        (self * 100.0).round() / 100.0
    }
}

const R_MIN: f32 = 0.70;
const R_MAX: f32 = 0.95;

#[derive(Debug, Clone, PartialEq)]
pub struct SimulatorConfig {
    pub deck_size: usize,
    pub learn_span: usize,
    pub max_cost_perday: f32,
    pub max_ivl: f32,
    pub learn_costs: [f32; 4],
    pub review_costs: [f32; 4],
    pub first_rating_prob: [f32; 4],
    pub review_rating_prob: [f32; 3],
    pub first_rating_offsets: [f32; 4],
    pub first_session_lens: [f32; 4],
    pub forget_rating_offset: f32,
    pub forget_session_len: f32,
    pub loss_aversion: f32,
    pub learn_limit: usize,
    pub review_limit: usize,
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            deck_size: 10000,
            learn_span: 365,
            max_cost_perday: 1800.0,
            max_ivl: 36500.0,
            learn_costs: [33.79, 24.3, 13.68, 6.5],
            review_costs: [23.0, 11.68, 7.33, 5.6],
            first_rating_prob: [0.24, 0.094, 0.495, 0.171],
            review_rating_prob: [0.224, 0.631, 0.145],
            first_rating_offsets: [-0.72, -0.15, -0.01, 0.0],
            first_session_lens: [2.02, 1.28, 0.81, 0.0],
            forget_rating_offset: -0.28,
            forget_session_len: 1.05,
            loss_aversion: 2.5,
            learn_limit: usize::MAX,
            review_limit: usize::MAX,
        }
    }
}

fn stability_after_success(w: &[f32], s: f32, r: f32, d: f32, rating: usize) -> f32 {
    let hard_penalty = if rating == 2 { w[15] } else { 1.0 };
    let easy_bonus = if rating == 4 { w[16] } else { 1.0 };
    (s * (f32::exp(w[8])
        * (11.0 - d)
        * s.powf(-w[9])
        * (f32::exp((1.0 - r) * w[10]) - 1.0)
        * hard_penalty)
        .mul_add(easy_bonus, 1.0))
    .clamp(S_MIN, S_MAX)
}

fn stability_after_failure(w: &[f32], s: f32, r: f32, d: f32) -> f32 {
    let new_s_min = s / (w[17] * w[18]).exp();
    let new_s =
        (w[11] * d.powf(-w[12]) * ((s + 1.0).powf(w[13]) - 1.0) * f32::exp((1.0 - r) * w[14]))
            .min(new_s_min);
    new_s.clamp(S_MIN, S_MAX)
}

fn stability_short_term(w: &[f32], s: f32, rating_offset: f32, session_len: f32) -> f32 {
    (s * (w[17] * (rating_offset + session_len * w[18])).exp()).clamp(S_MIN, S_MAX)
}

fn init_d(w: &[f32], rating: usize) -> f32 {
    w[4] - (w[5] * (rating - 1) as f32).exp() + 1.0
}

fn init_d_with_short_term(w: &[f32], rating: usize, rating_offset: f32) -> f32 {
    let new_d = init_d(w, rating) - w[6] * rating_offset;
    new_d.clamp(1.0, 10.0)
}

fn linear_damping(delta_d: f32, old_d: f32) -> f32 {
    (10.0 - old_d) / 9.0 * delta_d
}

fn next_d(w: &[f32], d: f32, rating: usize) -> f32 {
    let delta_d = -w[6] * (rating as f32 - 3.0);
    let new_d = d + linear_damping(delta_d, d);
    mean_reversion(w, init_d(w, 4), new_d).clamp(1.0, 10.0)
}

fn mean_reversion(w: &[f32], init: f32, current: f32) -> f32 {
    w[7] * init + (1.0 - w[7]) * current
}

fn power_forgetting_curve(t: f32, s: f32) -> f32 {
    (t / s).mul_add(FACTOR as f32, 1.0).powf(DECAY as f32)
}

#[derive(Debug, Clone)]
pub struct Card {
    pub difficulty: f32,
    pub stability: f32,
    pub last_date: f32,
    pub due: f32,
}

#[allow(clippy::type_complexity)]
pub fn simulate(
    config: &SimulatorConfig,
    w: &Parameters,
    desired_retention: f32,
    seed: Option<u64>,
    existing_cards: Option<Vec<Card>>,
) -> Result<(Array1<f32>, Array1<usize>, Array1<usize>, Array1<f32>), FSRSError> {
    let w = &check_and_fill_parameters(w)?;
    let w = &clip_parameters(w);
    let SimulatorConfig {
        deck_size,
        learn_span,
        max_cost_perday,
        max_ivl,
        learn_costs,
        review_costs,
        first_rating_prob,
        review_rating_prob,
        first_rating_offsets,
        first_session_lens,
        forget_rating_offset,
        forget_session_len,
        loss_aversion,
        learn_limit,
        review_limit,
    } = config.clone();
    if deck_size == 0 {
        return Err(FSRSError::InvalidDeckSize);
    }

    let mut review_cnt_per_day = Array1::<usize>::zeros(learn_span);
    let mut learn_cnt_per_day = Array1::<usize>::zeros(learn_span);
    let mut memorized_cnt_per_day = Array1::zeros(learn_span);
    let mut cost_per_day = Array1::zeros(learn_span);

    let first_rating_choices = [1, 2, 3, 4];
    let first_rating_dist = WeightedIndex::new(first_rating_prob).unwrap();

    let review_rating_choices = [2, 3, 4];
    let review_rating_dist = WeightedIndex::new(review_rating_prob).unwrap();

    let mut rng = StdRng::seed_from_u64(seed.unwrap_or(42));

    let mut cards = Vec::with_capacity(deck_size);

    let fail_cost = review_costs[0] * loss_aversion;

    let existing_count = if let Some(existing_cards) = &existing_cards {
        existing_cards.len()
    } else {
        0
    };

    if existing_count > deck_size {
        return Err(FSRSError::InvalidDeckSize);
    }

    if let Some(existing_cards) = existing_cards {
        cards.extend(
            existing_cards
                .into_iter()
                .filter(|card| card.stability > 1e-9),
        );

        for card in &cards {
            let upper = min(card.due as usize, learn_span);
            let elapsed_days = -card.last_date;
            for i in 0..upper {
                memorized_cnt_per_day[i] +=
                    power_forgetting_curve(elapsed_days + i as f32, card.stability);
            }
        }
    }

    if learn_limit > 0 {
        let init_ratings = (0..(deck_size - cards.len())).map(|i| Card {
            difficulty: f32::NEG_INFINITY,
            stability: f32::NEG_INFINITY,
            last_date: f32::NEG_INFINITY,
            due: (i / learn_limit) as f32,
        });

        cards.extend(init_ratings);
    }

    let mut card_priorities = PriorityQueue::new();

    fn card_priority(card: &Card, learn: bool) -> (usize, bool, usize) {
        // High difficulty priority as example
        (-card.due as usize, !learn, -card.difficulty as usize)
    }

    for (i, card) in cards.iter().enumerate() {
        card_priorities.push(i, card_priority(card, card.last_date == f32::NEG_INFINITY));
    }

    // Main simulation loop
    while let Some((&card_index, _)) = card_priorities.peek() {
        let card = &mut cards[card_index];

        let day_index = card.due as usize;

        let is_learn = card.last_date == f32::NEG_INFINITY;

        // Guards
        if card.due >= learn_span as f32 {
            card_priorities.pop();
            continue;
        }
        if (!is_learn && review_cnt_per_day[day_index] + 1 > review_limit)
            || (is_learn && learn_cnt_per_day[day_index] + 1 > learn_limit)
            || (cost_per_day[day_index] + fail_cost > max_cost_perday)
        {
            card.due += 1.;
            card_priorities.change_priority(&card_index, card_priority(card, is_learn));
            continue;
        }

        let ivl;

        // dbg!(&day_index);
        if is_learn {
            // For learning cards
            // Initialize stability and difficulty for new cards
            let rating = first_rating_choices[first_rating_dist.sample(&mut rng)];
            let offset = first_rating_offsets[rating - 1];

            card.difficulty = init_d_with_short_term(w, rating, offset);
            card.stability =
                stability_short_term(w, w[rating - 1], offset, first_session_lens[rating - 1]);

            ivl = next_interval(card.stability, desired_retention)
                .round()
                .clamp(1.0, max_ivl);

            // Update days statistics
            learn_cnt_per_day[day_index] += 1;
            cost_per_day[day_index] += learn_costs[rating - 1];
        } else {
            // For review cards
            // Updating delta_t for 'has_learned' cards
            let delta_t = card.due - card.last_date;

            // Calculate retrievability for entries where has_learned is true
            let retrievability = power_forgetting_curve(delta_t, card.stability);

            // Create 'forget' mask
            let forget = !rng.gen_bool(retrievability as f64);

            // Sample 'rating' for 'need_review' entries
            let rating = if forget {
                1
            } else {
                review_rating_choices[review_rating_dist.sample(&mut rng)]
            };

            //dbg!(&card, &rating);

            // Update stability
            card.stability = if forget {
                let post_lapse_stab =
                    stability_after_failure(w, card.stability, retrievability, card.difficulty);
                stability_short_term(w, post_lapse_stab, forget_rating_offset, forget_session_len)
            } else {
                stability_after_success(w, card.stability, retrievability, card.difficulty, rating)
            };

            // Update difficulty for review cards
            card.difficulty = next_d(w, card.difficulty, rating);
            if rating == 1 {
                card.difficulty -= (w[6] * forget_rating_offset).clamp(1.0, 10.0);
            }

            let cost = if forget {
                fail_cost
            } else {
                review_costs[rating - 1]
            };

            ivl = next_interval(card.stability, desired_retention)
                .round()
                .clamp(1.0, max_ivl);

            // Update days statistics
            review_cnt_per_day[day_index] += 1;
            cost_per_day[day_index] += cost;
        }

        let upper = min(ivl as usize, learn_span - day_index);
        for i in 0..upper {
            memorized_cnt_per_day[day_index + i] +=
                power_forgetting_curve(i as f32, card.stability);
        }

        // dbg!(ivl);

        // Update 'cost' based on 'forget' and 'rating'

        // dbg!(&review_cnt_per_day);

        // +1 because the day index is one less than the actual day as today is not graphed.
        card.last_date = card.due;
        card.due += ivl;

        if (card.due as usize) <= learn_span {
            card_priorities.change_priority(&card_index, card_priority(card, false));
        }
    }

    /*dbg!((
        &memorized_cnt_per_day[learn_span - 1],
        &review_cnt_per_day[learn_span - 1],
        &learn_cnt_per_day[learn_span - 1],
        &cost_per_day[learn_span - 1],
    ));*/

    Ok((
        memorized_cnt_per_day,
        review_cnt_per_day,
        learn_cnt_per_day,
        cost_per_day,
    ))
}

fn sample<F>(
    config: &SimulatorConfig,
    parameters: &Parameters,
    desired_retention: f32,
    n: usize,
    progress: &mut F,
) -> Result<f32, FSRSError>
where
    F: FnMut() -> bool,
{
    if !progress() {
        return Err(FSRSError::Interrupted);
    }
    let results: Result<Vec<f32>, FSRSError> = (0..n)
        .into_par_iter()
        .map(|i| {
            let (memorized_cnt_per_day, _, _, cost_per_day) = simulate(
                config,
                parameters,
                desired_retention,
                Some((i + 42).try_into().unwrap()),
                None,
            )?;
            let total_memorized = memorized_cnt_per_day[memorized_cnt_per_day.len() - 1];
            let total_cost = cost_per_day.sum();
            Ok(total_cost / total_memorized)
        })
        .collect();
    results.map(|v| v.iter().sum::<f32>() / n as f32)
}

impl<B: Backend> FSRS<B> {
    /// For the given simulator parameters and parameters, determine the suggested `desired_retention`
    /// value.
    pub fn optimal_retention<F>(
        &self,
        config: &SimulatorConfig,
        parameters: &Parameters,
        mut progress: F,
    ) -> Result<f32>
    where
        F: FnMut(ItemProgress) -> bool + Send,
    {
        let mut progress_info = ItemProgress {
            current: 0,
            // not provided for this method
            total: 0,
        };
        let inc_progress = move || {
            progress_info.current += 1;
            progress(progress_info)
        };

        Self::brent(config, parameters, inc_progress)
    }
    /// https://argmin-rs.github.io/argmin/argmin/solver/brent/index.html
    /// https://github.com/scipy/scipy/blob/5e4a5e3785f79dd4e8930eed883da89958860db2/scipy/optimize/_optimize.py#L2446
    fn brent<F>(
        config: &SimulatorConfig,
        parameters: &Parameters,
        mut progress: F,
    ) -> Result<f32, FSRSError>
    where
        F: FnMut() -> bool,
    {
        let mintol = 1e-10;
        let cg = 0.381_966;
        let maxiter = 64;
        let tol = 0.01f32;

        let default_sample_size = 16.0;
        let sample_size = match config.learn_span {
            ..=30 => 180,
            31..365 => {
                let (a1, a2, a3) = (8.20e-7, 2.41e-3, 1.30e-2);
                let factor = (config.learn_span as f32)
                    .powf(2.0)
                    .mul_add(a1, config.learn_span as f32 * a2 + a3);
                (default_sample_size / factor).round() as usize
            }
            365.. => default_sample_size as usize,
        };

        let (xb, fb) = (
            R_MIN,
            sample(config, parameters, R_MIN, sample_size, &mut progress)?,
        );
        let (mut x, mut v, mut w) = (xb, xb, xb);
        let (mut fx, mut fv, mut fw) = (fb, fb, fb);
        let (mut a, mut b) = (R_MIN, R_MAX);
        let mut deltax: f32 = 0.0;
        let mut iter = 0;
        let mut rat = 0.0;
        let mut u;

        while iter < maxiter {
            let tol1 = tol.mul_add(x.abs(), mintol);
            let tol2 = 2.0 * tol1;
            let xmid = 0.5 * (a + b);
            // check for convergence
            if (x - xmid).abs() < 0.5f32.mul_add(-(b - a), tol2) {
                break;
            }
            if deltax.abs() <= tol1 {
                // do a golden section step
                deltax = if x >= xmid { a } else { b } - x;
                rat = cg * deltax;
            } else {
                // do a parabolic step
                let tmp1 = (x - w) * (fx - fv);
                let mut tmp2 = (x - v) * (fx - fw);
                let mut p = (x - v).mul_add(tmp2, -(x - w) * tmp1);
                tmp2 = 2.0 * (tmp2 - tmp1);
                if tmp2 > 0.0 {
                    p = -p;
                }
                tmp2 = tmp2.abs();
                let deltax_tmp = deltax;
                deltax = rat;
                // check parabolic fit
                if (p > tmp2 * (a - x))
                    && (p < tmp2 * (b - x))
                    && (p.abs() < (0.5 * tmp2 * deltax_tmp).abs())
                {
                    // if parabolic step is useful
                    rat = p / tmp2;
                    u = x + rat;
                    if (u - a) < tol2 || (b - u) < tol2 {
                        rat = if xmid - x >= 0.0 { tol1 } else { -tol1 };
                    }
                } else {
                    // if it's not do a golden section step
                    deltax = if x >= xmid { a } else { b } - x;
                    rat = cg * deltax;
                }
            }
            // update by at least tol1
            u = x + if rat.abs() < tol1 {
                tol1 * if rat >= 0.0 { 1.0 } else { -1.0 }
            } else {
                rat
            };
            // calculate new output value
            let fu = sample(config, parameters, u, sample_size, &mut progress)?;

            // if it's bigger than current
            if fu > fx {
                if u < x {
                    a = u;
                } else {
                    b = u;
                }
                if fu <= fw || w == x {
                    (v, w) = (w, u);
                    (fv, fw) = (fw, fu);
                } else if fu <= fv || v == x || v == w {
                    v = u;
                    fv = fu;
                }
            } else {
                // if it's smaller than current
                if u >= x {
                    a = x;
                } else {
                    b = x;
                }
                (v, w, x) = (w, x, u);
                (fv, fw, fx) = (fw, fx, fu);
            }
            iter += 1;
        }
        let xmin = x;
        let success = iter < maxiter && (R_MIN..=R_MAX).contains(&xmin);
        dbg!(iter);

        if success {
            Ok(xmin)
        } else {
            Err(FSRSError::OptimalNotFound)
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum RevlogReviewKind {
    #[default]
    Learning = 0,
    Review = 1,
    Relearning = 2,
    /// Old Anki versions called this "Cram" or "Early", and assigned it when
    /// reviewing cards ahead. It is now only used for filtered decks with
    /// rescheduling disabled.
    Filtered = 3,
    Manual = 4,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RevlogEntry {
    pub id: i64,
    pub cid: i64,
    pub usn: i32,
    /// - In the V1 scheduler, 3 represents easy in the learning case.
    /// - 0 represents manual rescheduling.
    pub button_chosen: u8,
    /// Positive values are in days, negative values in seconds.
    pub interval: i32,
    /// Positive values are in days, negative values in seconds.
    pub last_interval: i32,
    /// Card's ease after answering, stored as 10x the %, eg 2500 represents
    /// 250%.
    pub ease_factor: u32,
    /// Amount of milliseconds taken to answer the card.
    pub taken_millis: u32,
    pub review_kind: RevlogReviewKind,
}

pub fn extract_simulator_config(
    df: Vec<RevlogEntry>,
    day_cutoff: i64,
    smooth: bool,
) -> SimulatorConfig {
    if df.is_empty() {
        return SimulatorConfig::default();
    }
    /*
        def rating_counts(x):
            tmp = defaultdict(int, x.value_counts().to_dict())
            first = x.iloc[0]
            tmp[first] -= 1
            return tmp
    */
    fn rating_counts(entries: &[RevlogEntry]) -> [u32; 4] {
        let mut counts = [0; 4];

        for entry in entries.iter().skip(1) {
            counts[entry.button_chosen as usize - 1] += 1;
        }

        counts
    }
    /*
        df1 = (
            df[(df["review_duration"] > 0) & (df["review_duration"] < 1200000)]
            .groupby(by=["card_id", "real_days"])
            .agg(
                {
                    "review_state": "first",
                    "review_rating": ["first", rating_counts],
                    "review_duration": "sum",
                }
            )
            .reset_index()
        )
    */
    struct Df1Row {
        card_id: i64,
        first_review_state: u8,
        first_review_rating: u8,
        review_rating_counts: [u32; 4],
        sum_review_duration: u32,
    }
    let df1 = {
        let mut grouped_data = HashMap::new();
        for &row in df.iter() {
            if row.taken_millis > 0 && row.taken_millis < 1200000 {
                let real_days = (row.id / 1000 - day_cutoff) / 86400;
                let key = (row.cid, real_days);
                grouped_data.entry(key).or_insert_with(Vec::new).push(row);
            }
        }

        grouped_data
            .into_iter()
            .filter_map(|((card_id, _real_days), entries)| {
                entries.first().map(|first_entry| {
                    let first_review_state = first_entry.review_kind as u8 + 1;
                    let first_review_rating = first_entry.button_chosen;
                    let review_rating_counts = rating_counts(&entries);
                    let sum_review_duration =
                        entries.iter().map(|entry| entry.taken_millis).sum::<u32>();

                    Df1Row {
                        card_id,
                        first_review_state,
                        first_review_rating,
                        review_rating_counts,
                        sum_review_duration,
                    }
                })
            })
            .collect_vec()
    };

    let cost_dict = {
        let mut cost_dict = HashMap::new();
        for row in df1.iter() {
            cost_dict
                .entry((row.first_review_state, row.first_review_rating))
                .or_insert_with(Vec::new)
                .push(row.sum_review_duration);
        }
        // calculate the median of the sum_review_duration
        fn median(x: &mut [u32]) -> u32 {
            x.sort_unstable();
            let n = x.len();
            if n % 2 == 0 {
                (x[n / 2 - 1] + x[n / 2]) / 2
            } else {
                x[n / 2]
            }
        }
        cost_dict
            .into_iter()
            .map(|(k, mut v)| (k, median(&mut v)))
            .collect::<HashMap<_, _>>()
    };

    // [cost_dict[(1, i)] / 1000 for i in range(1, 5)]
    let mut learn_costs: [f32; 4] = (1..5)
        .map(|i| cost_dict.get(&(1, i)).copied().unwrap_or_default() as f32 / 1000f32)
        .collect_vec()
        .try_into()
        .unwrap();
    // [cost_dict[(2, i)] / 1000 for i in range(1, 5)]
    let mut review_costs: [f32; 4] = (1..5)
        .map(|i| cost_dict.get(&(2, i)).copied().unwrap_or_default() as f32 / 1000f32)
        .collect_vec()
        .try_into()
        .unwrap();
    /*
        button_usage_dict = (
        df1.groupby(by=["first_review_state", "first_review_rating"])["card_id"]
        .count()
        .to_dict()
    ) */
    let button_usage_dict = {
        let mut button_usage_dict = HashMap::new();
        for row in df1.iter() {
            button_usage_dict
                .entry((row.first_review_state, row.first_review_rating))
                .or_insert_with(Vec::new)
                .push(row.card_id); // is this correct?
        }
        button_usage_dict
            .into_iter()
            .map(|(x, y)| (x, y.len() as i64))
            .collect::<HashMap<_, _>>()
    };
    // [button_usage_dict.get((1, i), 0) for i in range(1, 5)]
    let mut learn_buttons: [i64; 4] = (1..=4)
        .map(|i| button_usage_dict.get(&(1, i)).copied().unwrap_or_default())
        .collect_vec()
        .try_into()
        .unwrap();
    if learn_buttons.iter().all(|&x| x == 0) {
        learn_buttons = [1, 1, 1, 1];
    }
    // [button_usage_dict.get((2, i), 0) for i in range(1, 5)]
    let mut review_buttons: [i64; 4] = (1..=4)
        .map(|i| button_usage_dict.get(&(2, i)).copied().unwrap_or_default())
        .collect_vec()
        .try_into()
        .unwrap();
    if review_buttons.iter().skip(1).all(|&x| x == 0) {
        review_buttons = [review_buttons[0], 1, 1, 1];
    }
    // self.first_rating_prob = self.learn_buttons / self.learn_buttons.sum()
    let mut first_rating_prob: [f32; 4] = learn_buttons
        .iter()
        .map(|x| *x as f32 / learn_buttons.iter().sum::<i64>() as f32)
        .collect_vec()
        .try_into()
        .unwrap();
    // self.review_buttons[1:] / self.review_buttons[1:].sum()
    let mut review_rating_prob: [f32; 3] = review_buttons
        .iter()
        .skip(1)
        .map(|x| *x as f32 / review_buttons.iter().skip(1).sum::<i64>() as f32)
        .collect_vec()
        .try_into()
        .unwrap();

    // df2 = (
    //     df1.groupby(by=["first_review_state", "first_review_rating"])[[1, 2, 3, 4]]
    //     .mean()
    //     .round(2)
    // )

    let df2 = {
        let mut grouped = HashMap::new();
        for review in df1 {
            grouped
                .entry((review.first_review_state, review.first_review_rating))
                .or_insert_with(Vec::new)
                .push(review);
        }
        grouped
            .iter()
            .map(|((state, rating), group)| {
                let count = group.len() as f32;
                let (sum1, sum2, sum3, sum4) =
                    group
                        .iter()
                        .fold((0, 0, 0, 0), |(sum1, sum2, sum3, sum4), review| {
                            (
                                sum1 + review.review_rating_counts[0],
                                sum2 + review.review_rating_counts[1],
                                sum3 + review.review_rating_counts[2],
                                sum4 + review.review_rating_counts[3],
                            )
                        });

                let averages = [
                    (sum1 as f32 / count).to_2_decimal(),
                    (sum2 as f32 / count).to_2_decimal(),
                    (sum3 as f32 / count).to_2_decimal(),
                    (sum4 as f32 / count).to_2_decimal(),
                ];

                ((*state, *rating), averages)
            })
            .collect::<HashMap<_, _>>()
    };
    // rating_offset_dict = sum([df2[g] * (g - 3) for g in range(1, 5)]).to_dict()
    let rating_offset_dict = {
        let mut rating_offset_dict = HashMap::new();
        for (k, averages) in df2.iter() {
            let offset = averages
                .iter()
                .enumerate()
                .map(|(i, &v)| ((i + 1) as f32 - 3.0) * v)
                .sum::<f32>();
            rating_offset_dict.insert(k, (offset).to_2_decimal());
        }
        rating_offset_dict
    };
    // session_len_dict = sum([df2[g] for g in range(1, 5)]).to_dict()
    let session_len_dict = {
        let mut session_len_dict = HashMap::new();
        for (k, averages) in df2.iter() {
            let sum = averages.iter().sum::<f32>();
            session_len_dict.insert(k, (sum).to_2_decimal());
        }
        session_len_dict
    };
    // [rating_offset_dict[(1, i)] for i in range(1, 5)]
    let mut first_rating_offsets: [f32; 4] = (1..5)
        .map(|i| rating_offset_dict.get(&(1, i)).copied().unwrap_or_default())
        .collect_vec()
        .try_into()
        .unwrap();

    // [session_len_dict[(1, i)] for i in range(1, 5)]
    let mut first_session_lens: [f32; 4] = (1..5)
        .map(|i| session_len_dict.get(&(1, i)).copied().unwrap_or_default())
        .collect_vec()
        .try_into()
        .unwrap();

    first_rating_offsets[3] = 0.0;
    first_session_lens[3] = 0.0;

    // rating_offset_dict[(2, 1)]
    let mut forget_rating_offset = rating_offset_dict.get(&(2, 1)).copied().unwrap_or_default();
    // session_len_dict[(2, 1)]
    let mut forget_session_len = session_len_dict.get(&(2, 1)).copied().unwrap_or_default();
    ///  t * v0 + (1f32 - t) * v1
    fn lerp(v0: f32, v1: f32, t: f32) -> f32 {
        t * v0 + (1f32 - t) * v1
    }
    if smooth {
        let config = SimulatorConfig::default();

        izip!(
            &mut learn_costs,
            &mut first_rating_offsets,
            &mut first_session_lens,
            &learn_buttons,
            &config.learn_costs,
            &config.first_rating_offsets,
            &config.first_session_lens,
        )
        .for_each(
            |(
                learn_cost,
                first_rating_offset,
                first_session_len,
                &learn_button,
                &config_learn_cost,
                &config_first_rating_offset,
                &config_first_session_len,
            )| {
                let weight = learn_button as f32 / (50.0 + learn_button as f32);
                *learn_cost = lerp(*learn_cost, config_learn_cost, weight).to_2_decimal();
                *first_rating_offset =
                    lerp(*first_rating_offset, config_first_rating_offset, weight).to_2_decimal();
                *first_session_len =
                    lerp(*first_session_len, config_first_session_len, weight).to_2_decimal();
            },
        );

        let mut weight = [0.0f32; 4];
        izip!(
            &mut weight,
            &mut review_costs,
            &review_buttons,
            &config.review_costs
        )
        .for_each(
            |(weight, review_cost, review_button, config_review_costs)| {
                *weight = *review_button as f32 / (50.0 + *review_button as f32);
                *review_cost = lerp(*review_cost, *config_review_costs, *weight).to_2_decimal();
            },
        );

        forget_rating_offset =
            lerp(forget_rating_offset, config.forget_rating_offset, weight[0]).to_2_decimal();

        forget_session_len =
            lerp(forget_session_len, config.forget_session_len, weight[0]).to_2_decimal();

        let total_learn_buttons: i64 = learn_buttons.iter().sum();
        let weight = total_learn_buttons as f32 / (50.0 + total_learn_buttons as f32);
        first_rating_prob
            .iter_mut()
            .zip(config.first_rating_prob)
            .for_each(|(prob, first_rating_prob)| *prob = lerp(*prob, first_rating_prob, weight));
        let total_review_buttons_except_first: i64 = review_buttons[1..].iter().sum();
        let weight = total_review_buttons_except_first as f32
            / (50.0 + total_review_buttons_except_first as f32);

        review_rating_prob
            .iter_mut()
            .zip(config.review_rating_prob)
            .for_each(|(prob, review_rating_prob)| *prob = lerp(*prob, review_rating_prob, weight));
    }

    SimulatorConfig {
        learn_costs,
        review_costs,
        first_rating_prob,
        review_rating_prob,
        first_rating_offsets,
        first_session_lens,
        forget_rating_offset,
        forget_session_len,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DEFAULT_PARAMETERS, convertor_tests::read_collection};

    #[test]
    fn simulator() -> Result<()> {
        let config = SimulatorConfig::default();
        let (memorized_cnt_per_day, _, _, _) =
            simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, None)?;
        assert_eq!(
            memorized_cnt_per_day[memorized_cnt_per_day.len() - 1],
            7004.319
        );
        Ok(())
    }

    #[test]
    fn simulate_with_existing_cards() -> Result<()> {
        let config = SimulatorConfig {
            learn_span: 30,
            learn_limit: 60,
            review_limit: 200,
            max_cost_perday: f32::INFINITY,
            ..Default::default()
        };
        let cards = vec![
            Card {
                difficulty: 5.0,
                stability: 5.0,
                last_date: -5.0,
                due: 0.0,
            },
            Card {
                difficulty: 5.0,
                stability: 2.0,
                last_date: -2.0,
                due: 0.0,
            },
            Card {
                difficulty: 5.0,
                stability: 2.0,
                last_date: -2.0,
                due: 1.0,
            },
            Card {
                difficulty: 5.0,
                stability: 2.0,
                last_date: -8.0,
                due: -1.0,
            },
        ];
        let (memorized_cnt_per_day, review_cnt_per_day, learn_cnt_per_day, _) =
            simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, Some(cards))?;
        assert_eq!(memorized_cnt_per_day[0], 63.9);
        assert_eq!(review_cnt_per_day[0], 3);
        assert_eq!(learn_cnt_per_day[0], 60);
        Ok(())
    }

    #[test]
    fn simulate_with_learn_limit() -> Result<()> {
        let config = SimulatorConfig {
            learn_limit: 3,
            review_limit: 10,
            learn_span: 3,
            ..Default::default()
        };

        let cards = vec![
            Card {
                difficulty: 5.0,
                stability: 5.0,
                last_date: -5.0,
                due: 0.0,
            };
            9
        ];

        let (_, _, learn_cnt_per_day, _) =
            simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, Some(cards))?;

        assert_eq!(learn_cnt_per_day.to_vec(), vec![3, 3, 3]);

        Ok(())
    }

    #[test]
    fn simulator_learn_review_costs() -> Result<()> {
        const LEARN_COST: f32 = 42.;
        const REVIEW_COST: f32 = 43.;

        let config = SimulatorConfig {
            deck_size: 1, // 1 learn card, 1
            learn_costs: [LEARN_COST; 4],
            review_costs: [REVIEW_COST; 4],
            learn_span: 1,
            ..Default::default()
        };

        let cards = vec![Card {
            difficulty: 5.0,
            stability: 5.0,
            last_date: -5.0,
            due: 0.0,
        }];

        let (_, _, _, cost_per_day_learn) =
            simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, None)?;
        assert_eq!(cost_per_day_learn[0], LEARN_COST);
        let (_, _, _, cost_per_day_review) =
            simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, Some(cards))?;
        assert_eq!(cost_per_day_review[0], REVIEW_COST);
        Ok(())
    }

    #[test]
    fn simulate_with_learn_review_limit() -> Result<()> {
        let config = SimulatorConfig {
            learn_span: 30,
            learn_limit: 60,
            review_limit: 200,
            max_cost_perday: f32::INFINITY,
            ..Default::default()
        };
        let results = simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, None)?;
        assert_eq!(results.1.to_vec(), vec![
            0, 16, 25, 34, 60, 65, 76, 85, 91, 92, 100, 103, 119, 107, 103, 113, 122, 143, 149,
            151, 148, 172, 154, 175, 156, 169, 155, 191, 185, 170
        ]);
        assert_eq!(results.2.to_vec(), vec![
            config.learn_limit;
            config.learn_span
        ]);
        Ok(())
    }

    #[test]
    fn simulate_with_max_ivl() -> Result<()> {
        let config = SimulatorConfig {
            max_ivl: 100.0,
            ..Default::default()
        };
        let results = simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, None)?;
        assert_eq!(results.0[results.0.len() - 1], 6619.07);
        Ok(())
    }

    #[test]
    fn simulate_with_zero_card() -> Result<()> {
        let config = SimulatorConfig {
            deck_size: 0,
            ..Default::default()
        };
        let results = simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, None);
        assert_eq!(results.unwrap_err(), FSRSError::InvalidDeckSize);
        Ok(())
    }

    #[test]
    fn simulate_with_existing_cards_with_wrong_deck_size() -> Result<()> {
        let config = SimulatorConfig {
            deck_size: 1,
            ..Default::default()
        };
        let cards = vec![
            Card {
                difficulty: 5.0,
                stability: 5.0,
                last_date: -5.0,
                due: 0.0,
            },
            Card {
                difficulty: 5.0,
                stability: 2.0,
                last_date: -2.0,
                due: 0.0,
            },
        ];
        let results = simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, Some(cards));
        assert_eq!(results.unwrap_err(), FSRSError::InvalidDeckSize);
        Ok(())
    }

    #[test]
    fn optimal_retention() -> Result<()> {
        let learn_span = 1000;
        let learn_limit = 10;
        let fsrs = FSRS::new(None)?;
        let config = SimulatorConfig {
            deck_size: learn_span * learn_limit,
            learn_span,
            max_cost_perday: f32::INFINITY,
            learn_limit,
            ..Default::default()
        };
        let optimal_retention = fsrs.optimal_retention(&config, &[], |_v| true).unwrap();
        assert_eq!(optimal_retention, 0.8372485);
        assert!(fsrs.optimal_retention(&config, &[1.], |_v| true).is_err());
        Ok(())
    }

    #[test]
    fn optimal_retention_with_old_parameters() -> Result<()> {
        let learn_span = 1000;
        let learn_limit = 10;
        let fsrs = FSRS::new(None)?;
        let config = SimulatorConfig {
            deck_size: learn_span * learn_limit,
            learn_span,
            max_cost_perday: f32::INFINITY,
            learn_limit,
            ..Default::default()
        };
        let mut param = DEFAULT_PARAMETERS[..17].to_vec();
        param.extend_from_slice(&[0.0, 0.0]);
        let optimal_retention = fsrs.optimal_retention(&config, &param, |_v| true).unwrap();
        assert_eq!(optimal_retention, 0.85450846);
        Ok(())
    }

    #[test]
    fn extract_simulator_config_from_revlog() {
        let mut revlogs = read_collection().unwrap();
        revlogs.sort_by_cached_key(|r| (r.cid, r.id));
        let day_cutoff = 1720900800;
        let simulator_config = extract_simulator_config(revlogs.clone(), day_cutoff, false);
        assert_eq!(simulator_config, SimulatorConfig {
            learn_costs: [30.061, 0., 17.298, 12.352],
            review_costs: [19.139, 6.887, 5.83, 4.002],
            first_rating_prob: [0.19349411, 0., 0.14357824, 0.662_927_6],
            review_rating_prob: [0.07351815, 0.9011334, 0.025348445],
            first_rating_offsets: [1.64, 0., 0.69, 0.],
            first_session_lens: [2.74, 0., 1.32, 0.],
            forget_rating_offset: 1.28,
            forget_session_len: 1.77,
            ..Default::default()
        });

        let simulator_config = extract_simulator_config(revlogs, day_cutoff, true);
        assert_eq!(simulator_config, SimulatorConfig {
            learn_costs: [30.31, 24.3, 16.98, 12.23],
            review_costs: [19.37, 7.12, 5.84, 4.21],
            first_rating_prob: [0.19413717, 0.0012997796, 0.1484375, 0.65612555],
            review_rating_prob: [0.07409216, 0.900103, 0.025804851],
            first_rating_offsets: [1.48, -0.15, 0.63, 0.],
            first_session_lens: [2.69, 1.28, 1.27, 0.],
            forget_rating_offset: 1.19,
            forget_session_len: 1.73,
            ..Default::default()
        });
    }

    #[test]
    fn extract_simulator_config_without_revlog() {
        let simulator_config = extract_simulator_config(vec![], 0, true);
        assert_eq!(simulator_config, SimulatorConfig::default());
    }
}
