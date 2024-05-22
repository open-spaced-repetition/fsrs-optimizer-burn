use crate::error::{FSRSError, Result};
use crate::inference::{next_interval, ItemProgress, Parameters, DECAY, FACTOR, S_MIN};
use crate::{DEFAULT_PARAMETERS, FSRS};
use burn::tensor::backend::Backend;
use itertools::izip;
use ndarray::{s, Array1, Array2, Ix0, Ix1, SliceInfoElem, Zip};
use ndarray_rand::rand_distr::Distribution;
use ndarray_rand::RandomExt;
use rand::{
    distributions::{Uniform, WeightedIndex},
    rngs::StdRng,
    SeedableRng,
};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use strum::EnumCount;

#[derive(Debug, EnumCount)]
enum Column {
    Difficulty,
    Stability,
    #[allow(unused)]
    Retrievability,
    #[allow(unused)]
    DeltaT,
    LastDate,
    Due,
    Interval,
    #[allow(unused)]
    Cost,
    #[allow(unused)]
    Rand,
}

impl ndarray::SliceNextDim for Column {
    type InDim = Ix1;
    type OutDim = Ix0;
}

impl From<Column> for SliceInfoElem {
    fn from(value: Column) -> Self {
        Self::Index(value as isize)
    }
}

const R_MIN: f64 = 0.75;
const R_MAX: f64 = 0.95;

#[derive(Debug, Clone)]
pub struct SimulatorConfig {
    pub deck_size: usize,
    pub learn_span: usize,
    pub max_cost_perday: f64,
    pub max_ivl: f64,
    pub recall_costs: [f64; 3],
    pub forget_cost: f64,
    pub learn_cost: f64,
    pub first_rating_prob: [f64; 4],
    pub review_rating_prob: [f64; 3],
    pub loss_aversion: f64,
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
            recall_costs: [14.0, 10.0, 6.0],
            forget_cost: 50.0,
            learn_cost: 20.0,
            first_rating_prob: [0.15, 0.2, 0.6, 0.05],
            review_rating_prob: [0.3, 0.6, 0.1],
            loss_aversion: 2.5,
            learn_limit: usize::MAX,
            review_limit: usize::MAX,
        }
    }
}

fn stability_after_success(w: &[f64], s: f64, r: f64, d: f64, response: usize) -> f64 {
    let hard_penalty = if response == 2 { w[15] } else { 1.0 };
    let easy_bonus = if response == 4 { w[16] } else { 1.0 };
    s * (f64::exp(w[8])
        * (11.0 - d)
        * s.powf(-w[9])
        * (f64::exp((1.0 - r) * w[10]) - 1.0)
        * hard_penalty)
        .mul_add(easy_bonus, 1.0)
}

fn stability_after_failure(w: &[f64], s: f64, r: f64, d: f64) -> f64 {
    (w[11] * d.powf(-w[12]) * ((s + 1.0).powf(w[13]) - 1.0) * f64::exp((1.0 - r) * w[14]))
        .clamp(S_MIN.into(), s)
}

pub struct Card {
    pub difficulty: f64,
    pub stability: f64,
    pub last_date: f64,
    pub due: f64,
}

pub fn simulate(
    config: &SimulatorConfig,
    w: &[f64],
    desired_retention: f64,
    seed: Option<u64>,
    existing_cards: Option<Vec<Card>>,
) -> (Array1<f64>, Array1<usize>, Array1<usize>, Array1<f64>) {
    let SimulatorConfig {
        deck_size,
        learn_span,
        max_cost_perday,
        max_ivl,
        recall_costs,
        forget_cost,
        learn_cost,
        first_rating_prob,
        review_rating_prob,
        loss_aversion,
        learn_limit,
        review_limit,
    } = config.clone();
    let mut card_table = Array2::zeros((Column::COUNT, deck_size));
    card_table
        .slice_mut(s![Column::Due, ..])
        .fill(learn_span as f64);
    card_table.slice_mut(s![Column::Difficulty, ..]).fill(1e-10);
    card_table.slice_mut(s![Column::Stability, ..]).fill(1e-10);

    // fill card table based on existing_cards
    if let Some(existing_cards) = existing_cards {
        for (i, card) in existing_cards.into_iter().enumerate() {
            card_table[[Column::Difficulty as usize, i]] = card.difficulty;
            card_table[[Column::Stability as usize, i]] = card.stability;
            card_table[[Column::LastDate as usize, i]] = card.last_date;
            card_table[[Column::Due as usize, i]] = card.due;
        }
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

    // Main simulation loop
    for today in 0..learn_span {
        let old_stability = card_table.slice(s![Column::Stability, ..]);
        let has_learned = old_stability.mapv(|x| x > 1e-9);
        let old_last_date = card_table.slice(s![Column::LastDate, ..]);

        // Updating delta_t for 'has_learned' cards
        let mut delta_t = Array1::zeros(deck_size); // Create an array of the same length for delta_t

        // Calculate delta_t for entries where has_learned is true
        izip!(&mut delta_t, &old_last_date, &has_learned)
            .filter(|(.., &has_learned_flag)| has_learned_flag)
            .for_each(|(delta_t, &last_date, ..)| {
                *delta_t = today as f64 - last_date;
            });

        let mut retrievability = Array1::zeros(deck_size); // Create an array for retrievability

        fn power_forgetting_curve(t: f64, s: f64) -> f64 {
            (t / s).mul_add(FACTOR, 1.0).powf(DECAY)
        }

        // Calculate retrievability for entries where has_learned is true
        izip!(&mut retrievability, &delta_t, &old_stability, &has_learned)
            .filter(|(.., &has_learned_flag)| has_learned_flag)
            .for_each(|(retrievability, &delta_t, &stability, ..)| {
                *retrievability = power_forgetting_curve(delta_t, stability)
            });

        // Set 'cost' column to 0
        let mut cost = Array1::<f64>::zeros(deck_size);

        // Create 'need_review' mask
        let old_due = card_table.slice(s![Column::Due, ..]);
        let need_review = old_due.mapv(|x| x <= today as f64);

        // dbg!(&need_review.mapv(|x| x as i32).sum());

        // Update 'rand' column for 'need_review' entries
        let mut rand_slice = Array1::zeros(deck_size);
        let n_need_review = need_review.iter().filter(|&&x| x).count();
        let random_values = Array1::random_using(n_need_review, Uniform::new(0.0, 1.0), &mut rng);

        rand_slice
            .iter_mut()
            .zip(&need_review)
            .filter(|(_, &need_review_flag)| need_review_flag)
            .map(|(x, _)| x)
            .zip(random_values)
            .for_each(|(rand_elem, random_value)| {
                *rand_elem = random_value;
            });

        // Create 'forget' mask
        let forget = Zip::from(&rand_slice)
            .and(&retrievability)
            .map_collect(|&rand_val, &retriev_val| rand_val > retriev_val);

        // Sample 'rating' for 'need_review' entries
        let mut ratings = Array1::zeros(deck_size);
        izip!(&mut ratings, &(&need_review & !&forget))
            .filter(|(_, &condition)| condition)
            .for_each(|(rating, _)| {
                *rating = review_rating_choices[review_rating_dist.sample(&mut rng)]
            });

        // Update 'cost' column based on 'need_review', 'forget' and 'ratings'
        izip!(&mut cost, &need_review, &forget, &ratings)
            .filter(|(_, &need_review_flag, _, _)| need_review_flag)
            .for_each(|(cost, _, &forget_flag, &rating)| {
                *cost = if forget_flag {
                    forget_cost * loss_aversion
                } else {
                    recall_costs[rating - 2]
                }
            });

        // Calculate cumulative sum of 'cost'
        let mut cum_sum = Array1::<f64>::zeros(deck_size);
        cum_sum[0] = cost[0];
        for i in 1..deck_size {
            cum_sum[i] = cum_sum[i - 1] + cost[i];
        }

        // Create 'true_review' mask based on 'need_review' and 'cum_sum' and 'review_limit'
        let mut review_count = 0;
        let true_review =
            Zip::from(&need_review)
                .and(&cum_sum)
                .map_collect(|&need_review_flag, &cum_cost| {
                    if need_review_flag {
                        review_count += 1;
                    }
                    need_review_flag
                        && (cum_cost <= max_cost_perday)
                        && (review_count <= review_limit)
                });

        let need_learn = old_due.mapv(|x| x == learn_span as f64);
        // Update 'cost' column based on 'need_learn'
        izip!(&mut cost, &need_learn)
            .filter(|(_, &need_learn_flag)| need_learn_flag)
            .for_each(|(cost, _)| {
                *cost = learn_cost;
            });

        cum_sum[0] = cost[0];
        for i in 1..deck_size {
            cum_sum[i] = cum_sum[i - 1] + cost[i];
        }

        // dbg!(&cum_sum);

        // Create 'true_learn' mask based on 'need_learn' and 'cum_sum' and 'learn_limit'
        let mut learn_count = 0;
        let true_learn =
            Zip::from(&need_learn)
                .and(&cum_sum)
                .map_collect(|&need_learn_flag, &cum_cost| {
                    if need_learn_flag {
                        learn_count += 1;
                    }
                    need_learn_flag && (cum_cost <= max_cost_perday) && (learn_count <= learn_limit)
                });

        // Sample 'rating' for 'true_learn' entries
        izip!(&mut ratings, &true_learn)
            .filter(|(_, &true_learn_flag)| true_learn_flag)
            .for_each(|(rating, _)| {
                *rating = first_rating_choices[first_rating_dist.sample(&mut rng)]
            });

        let mut new_stability = old_stability.to_owned();
        let old_difficulty = card_table.slice(s![Column::Difficulty, ..]);
        // Iterate over slices and apply stability_after_failure function
        izip!(
            &mut new_stability,
            &old_stability,
            &retrievability,
            &old_difficulty,
            &(&true_review & &forget)
        )
        .filter(|(.., &condition)| condition)
        .for_each(|(new_stab, &stab, &retr, &diff, ..)| {
            *new_stab = stability_after_failure(w, stab, retr, diff);
        });

        // Iterate over slices and apply stability_after_success function
        izip!(
            &mut new_stability,
            &ratings,
            &old_stability,
            &retrievability,
            &old_difficulty,
            &(&true_review & !&forget)
        )
        .filter(|(.., &condition)| condition)
        .for_each(|(new_stab, &rating, &stab, &retr, &diff, _)| {
            *new_stab = stability_after_success(w, stab, retr, diff, rating);
        });

        // Initialize a new Array1 to store updated difficulty values
        let mut new_difficulty = old_difficulty.to_owned();

        // Update the difficulty values based on the condition 'true_review & forget'
        izip!(&mut new_difficulty, &old_difficulty, &true_review, &forget)
            .filter(|(.., &true_rev, &frgt)| true_rev && frgt)
            .for_each(|(new_diff, &old_diff, ..)| {
                *new_diff = (2.0f64.mul_add(w[6], old_diff)).clamp(1.0, 10.0);
            });

        // Update the difficulty values based on the condition 'true_review & !forget'
        izip!(
            &mut new_difficulty,
            &old_difficulty,
            &ratings,
            &(&true_review & !&forget)
        )
        .filter(|(.., &condition)| condition)
        .for_each(|(new_diff, &old_diff, &rating, ..)| {
            *new_diff = w[6].mul_add(3.0 - rating as f64, old_diff).clamp(1.0, 10.0);
        });

        // Update 'last_date' column where 'true_review' or 'true_learn' is true
        let mut new_last_date = old_last_date.to_owned();
        izip!(&mut new_last_date, &true_review, &true_learn)
            .filter(|(_, &true_review_flag, &true_learn_flag)| true_review_flag || true_learn_flag)
            .for_each(|(new_last_date, ..)| {
                *new_last_date = today as f64;
            });

        izip!(
            &mut new_stability,
            &mut new_difficulty,
            &ratings,
            &true_learn
        )
        .filter(|(.., &true_learn_flag)| true_learn_flag)
        .for_each(|(new_stab, new_diff, &rating, _)| {
            *new_stab = w[rating - 1];
            *new_diff = (w[5].mul_add(-(rating as f64 - 3.0), w[4])).clamp(1.0, 10.0);
        });
        let old_interval = card_table.slice(s![Column::Interval, ..]);
        let mut new_interval = old_interval.to_owned();
        izip!(&mut new_interval, &new_stability, &true_review, &true_learn)
            .filter(|(.., &true_review_flag, &true_learn_flag)| true_review_flag || true_learn_flag)
            .for_each(|(new_ivl, &new_stab, ..)| {
                *new_ivl = (next_interval(new_stab as f32, desired_retention as f32) as f64)
                    .clamp(1.0, max_ivl);
            });

        let old_due = card_table.slice(s![Column::Due, ..]);
        let mut new_due = old_due.to_owned();
        izip!(&mut new_due, &new_interval, &true_review, &true_learn)
            .filter(|(.., &true_review_flag, &true_learn_flag)| true_review_flag || true_learn_flag)
            .for_each(|(new_due, &new_ivl, ..)| {
                *new_due = today as f64 + new_ivl;
            });

        // Update the card_table with the new values
        card_table
            .slice_mut(s![Column::Difficulty, ..])
            .assign(&new_difficulty);
        card_table
            .slice_mut(s![Column::Stability, ..])
            .assign(&new_stability);
        card_table
            .slice_mut(s![Column::LastDate, ..])
            .assign(&new_last_date);
        card_table.slice_mut(s![Column::Due, ..]).assign(&new_due);
        card_table
            .slice_mut(s![Column::Interval, ..])
            .assign(&new_interval);
        // Update the review_cnt_per_day, learn_cnt_per_day and memorized_cnt_per_day
        review_cnt_per_day[today] = true_review.iter().filter(|&&x| x).count();
        learn_cnt_per_day[today] = true_learn.iter().filter(|&&x| x).count();
        memorized_cnt_per_day[today] = retrievability.sum();
        cost_per_day[today] = izip!(cost, &true_review, &true_learn)
            .filter(|(_, &true_review_flag, &true_learn_flag)| true_review_flag || true_learn_flag)
            .map(|(cost, ..)| cost)
            .sum();
    }

    (
        memorized_cnt_per_day,
        review_cnt_per_day,
        learn_cnt_per_day,
        cost_per_day,
    )
}

fn sample<F>(
    config: &SimulatorConfig,
    parameters: &[f64],
    desired_retention: f64,
    n: usize,
    progress: &mut F,
) -> Result<f64>
where
    F: FnMut() -> bool,
{
    if !progress() {
        return Err(FSRSError::Interrupted);
    }
    Ok((0..n)
        .into_par_iter()
        .map(|i| {
            let (memorized_cnt_per_day, _, _, cost_per_day) = simulate(
                config,
                parameters,
                desired_retention,
                Some((i + 42).try_into().unwrap()),
                None,
            );
            let total_memorized = memorized_cnt_per_day[memorized_cnt_per_day.len() - 1];
            let total_cost = cost_per_day.sum();
            total_cost / total_memorized
        })
        .sum::<f64>()
        / n as f64)
}

const SAMPLE_SIZE: usize = 4;

impl<B: Backend> FSRS<B> {
    /// For the given simulator parameters and parameters, determine the suggested `desired_retention`
    /// value.
    pub fn optimal_retention<F>(
        &self,
        config: &SimulatorConfig,
        parameters: &Parameters,
        mut progress: F,
    ) -> Result<f64>
    where
        F: FnMut(ItemProgress) -> bool + Send,
    {
        let parameters = if parameters.is_empty() {
            &DEFAULT_PARAMETERS
        } else if parameters.len() != 17 {
            return Err(FSRSError::InvalidParameters);
        } else {
            parameters
        }
        .iter()
        .map(|v| *v as f64)
        .collect::<Vec<_>>();
        let mut progress_info = ItemProgress {
            current: 0,
            // not provided for this method
            total: 0,
        };
        let inc_progress = move || {
            progress_info.current += 1;
            progress(progress_info)
        };

        Self::brent(config, &parameters, inc_progress)
    }
    /// https://argmin-rs.github.io/argmin/argmin/solver/brent/index.html
    /// https://github.com/scipy/scipy/blob/5e4a5e3785f79dd4e8930eed883da89958860db2/scipy/optimize/_optimize.py#L2446
    fn brent<F>(
        config: &SimulatorConfig,
        parameters: &[f64],
        mut progress: F,
    ) -> Result<f64, FSRSError>
    where
        F: FnMut() -> bool,
    {
        let mintol = 1e-10;
        let cg = 0.3819660;
        let maxiter = 64;
        let tol = 0.01f64;

        let (xb, fb) = (
            R_MIN,
            sample(config, parameters, R_MIN, SAMPLE_SIZE, &mut progress)?,
        );
        let (mut x, mut v, mut w) = (xb, xb, xb);
        let (mut fx, mut fv, mut fw) = (fb, fb, fb);
        let (mut a, mut b) = (R_MIN, R_MAX);
        let mut deltax: f64 = 0.0;
        let mut iter = 0;
        let mut rat = 0.0;
        let mut u;

        while iter < maxiter {
            let tol1 = tol.mul_add(x.abs(), mintol);
            let tol2 = 2.0 * tol1;
            let xmid = 0.5 * (a + b);
            // check for convergence
            if (x - xmid).abs() < 0.5f64.mul_add(-(b - a), tol2) {
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
            let fu = sample(config, parameters, u, SAMPLE_SIZE, &mut progress)?;

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

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::*;
    use crate::DEFAULT_PARAMETERS;

    #[test]
    fn simulator() {
        let config = SimulatorConfig::default();
        let (memorized_cnt_per_day, _, _, _) = simulate(
            &config,
            &DEFAULT_PARAMETERS.iter().map(|v| *v as f64).collect_vec(),
            0.9,
            None,
            None,
        );
        assert_eq!(
            memorized_cnt_per_day[memorized_cnt_per_day.len() - 1],
            3199.9526251977177
        )
    }

    #[test]
    fn simulate_with_existing_cards() {
        let config = SimulatorConfig {
            learn_span: 30,
            learn_limit: 60,
            review_limit: 200,
            max_cost_perday: f64::INFINITY,
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
        let memorization = simulate(
            &config,
            &DEFAULT_PARAMETERS.iter().map(|v| *v as f64).collect_vec(),
            0.9,
            None,
            Some(cards),
        );
        dbg!(memorization);
    }

    #[test]
    fn simulate_with_learn_review_limit() {
        let config = SimulatorConfig {
            learn_span: 30,
            learn_limit: 60,
            review_limit: 200,
            max_cost_perday: f64::INFINITY,
            ..Default::default()
        };
        let results = simulate(
            &config,
            &DEFAULT_PARAMETERS.iter().map(|v| *v as f64).collect_vec(),
            0.9,
            None,
            None,
        );
        assert_eq!(
            results.1.to_vec(),
            vec![
                0, 16, 27, 34, 84, 80, 91, 92, 104, 106, 109, 112, 133, 123, 139, 121, 136, 149,
                136, 159, 173, 178, 175, 180, 189, 181, 196, 200, 193, 196
            ]
        );
        assert_eq!(
            results.2.to_vec(),
            vec![config.learn_limit; config.learn_span]
        )
    }

    #[test]
    fn optimal_retention() -> Result<()> {
        let learn_span = 1000;
        let learn_limit = 10;
        let fsrs = FSRS::new(None)?;
        let config = SimulatorConfig {
            deck_size: learn_span * learn_limit,
            learn_span,
            max_cost_perday: f64::INFINITY,
            learn_limit,
            loss_aversion: 2.5,
            ..Default::default()
        };
        let optimal_retention = fsrs.optimal_retention(&config, &[], |_v| true).unwrap();
        assert_eq!(optimal_retention, 0.8419900928572013);
        assert!(fsrs.optimal_retention(&config, &[1.], |_v| true).is_err());
        Ok(())
    }
}
