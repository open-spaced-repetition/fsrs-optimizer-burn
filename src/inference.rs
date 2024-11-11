use std::collections::HashMap;
use std::ops::{Add, Sub};

use crate::model::{Get, MemoryStateTensors, FSRS};
use burn::nn::loss::Reduction;
use burn::tensor::{Data, Shape, Tensor};
use burn::{data::dataloader::batcher::Batcher, tensor::backend::Backend};

use crate::dataset::FSRSBatch;
use crate::dataset::FSRSBatcher;
use crate::error::Result;
use crate::model::Model;
use crate::training::BCELoss;
use crate::{FSRSError, FSRSItem};
use burn::tensor::ElementConversion;
pub(crate) const DECAY: f64 = -0.5;
/// (9/10) ^ (1 / DECAY) - 1
pub(crate) const FACTOR: f64 = 19f64 / 81f64;
pub(crate) const S_MIN: f32 = 0.01;
pub(crate) const S_MAX: f32 = 36500.0;
pub(crate) const D_MIN: f32 = 1.0;
pub(crate) const D_MAX: f32 = 10.0;
/// This is a slice for efficiency, but should always be 17 in length.
pub type Parameters = [f32];
use itertools::izip;

pub static DEFAULT_PARAMETERS: [f32; 19] = [
    0.40255, 1.18385, 3.173, 15.69105, 7.1949, 0.5345, 1.4604, 0.0046, 1.54575, 0.1192, 1.01925,
    1.9395, 0.11, 0.29605, 2.2698, 0.2315, 2.9898, 0.51655, 0.6621,
];

fn infer<B: Backend>(
    model: &Model<B>,
    batch: FSRSBatch<B>,
) -> (MemoryStateTensors<B>, Tensor<B, 1>) {
    let state = model.forward(batch.t_historys, batch.r_historys, None);
    let retention = model.power_forgetting_curve(batch.delta_ts, state.stability.clone());
    (state, retention)
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct MemoryState {
    pub stability: f32,
    pub difficulty: f32,
}

impl<B: Backend> From<MemoryStateTensors<B>> for MemoryState {
    fn from(m: MemoryStateTensors<B>) -> Self {
        Self {
            stability: m.stability.to_data().value[0].elem(),
            difficulty: m.difficulty.to_data().value[0].elem(),
        }
    }
}

impl<B: Backend> From<MemoryState> for MemoryStateTensors<B> {
    fn from(m: MemoryState) -> Self {
        Self {
            stability: Tensor::from_data(
                Data::new(vec![m.stability.elem()], Shape { dims: [1] }),
                &B::Device::default(),
            ),
            difficulty: Tensor::from_data(
                Data::new(vec![m.difficulty.elem()], Shape { dims: [1] }),
                &B::Device::default(),
            ),
        }
    }
}

pub fn next_interval(stability: f32, desired_retention: f32) -> f32 {
    stability / FACTOR as f32 * (desired_retention.powf(1.0 / DECAY as f32) - 1.0)
}

impl<B: Backend> FSRS<B> {
    /// Calculate the current memory state for a given card's history of reviews.
    /// In the case of truncated reviews, [starting_state] can be set to the value of
    /// [FSRS::memory_state_from_sm2] for the first review (which should not be included
    /// in FSRSItem). If not provided, the card starts as new.
    /// Parameters must have been provided when calling FSRS::new().
    pub fn memory_state(
        &self,
        item: FSRSItem,
        starting_state: Option<MemoryState>,
    ) -> Result<MemoryState> {
        let (time_history, rating_history) =
            item.reviews.iter().map(|r| (r.delta_t, r.rating)).unzip();
        let size = item.reviews.len();
        let time_history = Tensor::from_data(
            Data::new(time_history, Shape { dims: [size] }).convert(),
            &self.device(),
        )
        .unsqueeze()
        .transpose();
        let rating_history = Tensor::from_data(
            Data::new(rating_history, Shape { dims: [size] }).convert(),
            &self.device(),
        )
        .unsqueeze()
        .transpose();
        let state: MemoryState = self
            .model()
            .forward(time_history, rating_history, starting_state.map(Into::into))
            .into();
        if !state.stability.is_finite() || !state.difficulty.is_finite() {
            Err(FSRSError::InvalidInput)
        } else {
            Ok(state)
        }
    }

    /// If a card has incomplete learning history, memory state can be approximated from
    /// current sm2 values.
    /// Parameters must have been provided when calling FSRS::new().
    pub fn memory_state_from_sm2(
        &self,
        ease_factor: f32,
        interval: f32,
        sm2_retention: f32,
    ) -> Result<MemoryState> {
        let stability =
            interval.max(S_MIN) * FACTOR as f32 / (sm2_retention.powf(1.0 / DECAY as f32) - 1.0);
        let w = &self.model().w;
        let w8: f32 = w.get(8).into_scalar().elem();
        let w9: f32 = w.get(9).into_scalar().elem();
        let w10: f32 = w.get(10).into_scalar().elem();
        let difficulty = 11.0
            - (ease_factor - 1.0)
                / (w8.exp() * stability.powf(-w9) * ((1.0 - sm2_retention) * w10).exp_m1());
        if !stability.is_finite() || !difficulty.is_finite() {
            Err(FSRSError::InvalidInput)
        } else {
            Ok(MemoryState {
                stability,
                difficulty: difficulty.clamp(D_MIN, D_MAX),
            })
        }
    }

    /// Calculate the next interval for the current memory state, for rescheduling. Stability
    /// should be provided except when the card is new. Rating is ignored except when card is new.
    /// Parameters must have been provided when calling FSRS::new().
    pub fn next_interval(
        &self,
        stability: Option<f32>,
        desired_retention: f32,
        rating: u32,
    ) -> f32 {
        let stability = stability.unwrap_or_else(|| {
            // get initial stability for new card
            let rating = Tensor::from_data(
                Data::new(vec![rating.elem()], Shape { dims: [1] }),
                &self.device(),
            );
            let model = self.model();
            model.init_stability(rating).into_scalar().elem()
        });
        next_interval(stability, desired_retention)
    }

    /// The intervals and memory states for each answer button.
    /// Parameters must have been provided when calling FSRS::new().
    pub fn next_states(
        &self,
        current_memory_state: Option<MemoryState>,
        desired_retention: f32,
        days_elapsed: u32,
    ) -> Result<NextStates> {
        let delta_t = Tensor::from_data(
            Data::new(vec![days_elapsed.elem()], Shape { dims: [1] }),
            &self.device(),
        );
        let current_memory_state_tensors = current_memory_state.map(MemoryStateTensors::from);
        let model = self.model();
        let mut next_memory_states = (1..=4).map(|rating| {
            Ok({
                let state = MemoryState::from(model.step(
                    delta_t.clone(),
                    Tensor::from_data(
                        Data::new(vec![rating.elem()], Shape { dims: [1] }),
                        &self.device(),
                    ),
                    current_memory_state_tensors.clone(),
                ));
                if !state.stability.is_finite() || !state.difficulty.is_finite() {
                    return Err(FSRSError::InvalidInput);
                }
                state
            })
        });

        let mut get_next_state = || {
            let memory = next_memory_states.next().unwrap()?;
            let interval = next_interval(memory.stability, desired_retention);
            Ok(ItemState { memory, interval })
        };

        Ok(NextStates {
            again: get_next_state()?,
            hard: get_next_state()?,
            good: get_next_state()?,
            easy: get_next_state()?,
        })
    }

    /// Determine how well the model and parameters predict performance.
    /// Parameters must have been provided when calling FSRS::new().
    pub fn evaluate<F>(&self, items: Vec<FSRSItem>, mut progress: F) -> Result<ModelEvaluation>
    where
        F: FnMut(ItemProgress) -> bool,
    {
        if items.is_empty() {
            return Err(FSRSError::NotEnoughData);
        }
        let batcher = FSRSBatcher::new(self.device());
        let mut all_retention = vec![];
        let mut all_labels = vec![];
        let mut progress_info = ItemProgress {
            current: 0,
            total: items.len(),
        };
        let model = self.model();
        let mut r_matrix: HashMap<(u32, u32, u32), (f32, f32, f32)> = HashMap::new();

        for chunk in items.chunks(512) {
            let batch = batcher.batch(chunk.to_vec());
            let (_state, retention) = infer::<B>(model, batch.clone());
            let pred = retention.clone().to_data().convert::<f32>().value;
            let true_val = batch.labels.clone().to_data().convert::<f32>().value;
            all_retention.push(retention);
            all_labels.push(batch.labels);
            izip!(chunk, pred, true_val).for_each(|(item, p, y)| {
                let bin = item.r_matrix_index();
                let (pred, real, count) = r_matrix.entry(bin).or_insert((0.0, 0.0, 0.0));
                *pred += p;
                *real += y;
                *count += 1.0;
            });
            progress_info.current += chunk.len();
            if !progress(progress_info) {
                return Err(FSRSError::Interrupted);
            }
        }
        let rmse = (r_matrix
            .values()
            .map(|(pred, real, count)| {
                let pred = pred / count;
                let real = real / count;
                (pred - real).powi(2) * count
            })
            .sum::<f32>()
            / r_matrix.values().map(|(_, _, count)| count).sum::<f32>())
        .sqrt();
        let all_retention = Tensor::cat(all_retention, 0);
        let all_labels = Tensor::cat(all_labels, 0).float();
        let loss = BCELoss::new().forward(all_retention, all_labels, Reduction::Mean);
        Ok(ModelEvaluation {
            log_loss: loss.to_data().value[0].elem(),
            rmse_bins: rmse,
        })
    }

    /// How well the user is likely to remember the item after `days_elapsed` since the previous
    /// review.
    pub fn current_retrievability(&self, state: MemoryState, days_elapsed: u32) -> f32 {
        (days_elapsed as f64 / state.stability as f64 * FACTOR + 1.0).powf(DECAY) as f32
    }

    /// Returns the universal metrics for the existing and provided parameters. If the first value
    /// is smaller than the second value, the existing parameters are better than the provided ones.
    pub fn universal_metrics<F>(
        &self,
        items: Vec<FSRSItem>,
        parameters: &Parameters,
        mut progress: F,
    ) -> Result<(f32, f32)>
    where
        F: FnMut(ItemProgress) -> bool,
    {
        if items.is_empty() {
            return Err(FSRSError::NotEnoughData);
        }
        let batcher = FSRSBatcher::new(self.device());
        let mut all_predictions_self = vec![];
        let mut all_predictions_other = vec![];
        let mut all_true_val = vec![];
        let mut progress_info = ItemProgress {
            current: 0,
            total: items.len(),
        };
        let model_self = self.model();
        let fsrs_other = Self::new_with_backend(Some(parameters), self.device())?;
        let model_other = fsrs_other.model();
        for chunk in items.chunks(512) {
            let batch = batcher.batch(chunk.to_vec());

            let (_state, retention) = infer::<B>(model_self, batch.clone());
            let pred = retention.clone().to_data().convert::<f32>().value;
            all_predictions_self.extend(pred);

            let (_state, retention) = infer::<B>(model_other, batch.clone());
            let pred = retention.clone().to_data().convert::<f32>().value;
            all_predictions_other.extend(pred);

            let true_val = batch.labels.clone().to_data().convert::<f32>().value;
            all_true_val.extend(true_val);
            progress_info.current += chunk.len();
            if !progress(progress_info) {
                return Err(FSRSError::Interrupted);
            }
        }
        let self_by_other =
            measure_a_by_b(&all_predictions_self, &all_predictions_other, &all_true_val);
        let other_by_self =
            measure_a_by_b(&all_predictions_other, &all_predictions_self, &all_true_val);
        Ok((self_by_other, other_by_self))
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ModelEvaluation {
    pub log_loss: f32,
    pub rmse_bins: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NextStates {
    pub again: ItemState,
    pub hard: ItemState,
    pub good: ItemState,
    pub easy: ItemState,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ItemState {
    pub memory: MemoryState,
    pub interval: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct ItemProgress {
    pub current: usize,
    pub total: usize,
}

fn get_bin(x: f32, bins: i32) -> i32 {
    let log_base = (bins.add(1) as f32).ln();
    let binned_x = (x * log_base).exp().floor().sub(1.0);
    (binned_x as i32).clamp(0, bins - 1)
}

fn measure_a_by_b(pred_a: &[f32], pred_b: &[f32], true_val: &[f32]) -> f32 {
    let mut groups = HashMap::new();
    izip!(pred_a, pred_b, true_val).for_each(|(a, b, t)| {
        let bin = get_bin(*b, 20);
        groups.entry(bin).or_insert_with(Vec::new).push((a, t));
    });
    let mut total_sum = 0.0;
    let mut total_count = 0.0;
    for group in groups.values() {
        let count = group.len() as f32;
        let pred_mean = group.iter().map(|(p, _)| *p).sum::<f32>() / count;
        let true_mean = group.iter().map(|(_, t)| *t).sum::<f32>() / count;

        let rmse = (pred_mean - true_mean).powi(2);
        total_sum += rmse * count;
        total_count += count;
    }

    (total_sum / total_count).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        convertor_tests::anki21_sample_file_converted_to_fsrs, dataset::filter_outlier, FSRSReview,
    };

    static PARAMETERS: &[f32] = &[
        0.6845422,
        1.6790825,
        4.7349424,
        10.042885,
        7.4410233,
        0.64219797,
        1.071918,
        0.0025195254,
        1.432437,
        0.1544,
        0.8692766,
        2.0696752,
        0.0953,
        0.2975,
        2.4691248,
        0.19542035,
        3.201072,
        0.18046261,
        0.121442534,
    ];
    fn assert_approx_eq(a: [f32; 2], b: [f32; 2]) {
        Data::from(a).assert_approx_eq(&Data::from(b), 5);
    }
    #[test]
    fn test_get_bin() {
        let pred = (0..=100).map(|i| i as f32 / 100.0).collect::<Vec<_>>();
        let bin = pred.iter().map(|p| get_bin(*p, 20)).collect::<Vec<_>>();
        assert_eq!(
            bin,
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4,
                4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
                11, 11, 11, 12, 12, 13, 13, 14, 14, 14, 15, 15, 16, 17, 17, 18, 18, 19, 19
            ]
        );
    }

    #[test]
    fn test_memo_state() -> Result<()> {
        let item = FSRSItem {
            reviews: vec![
                FSRSReview {
                    rating: 1,
                    delta_t: 0,
                },
                FSRSReview {
                    rating: 3,
                    delta_t: 1,
                },
                FSRSReview {
                    rating: 3,
                    delta_t: 3,
                },
                FSRSReview {
                    rating: 3,
                    delta_t: 8,
                },
                FSRSReview {
                    rating: 3,
                    delta_t: 21,
                },
            ],
        };
        let fsrs = FSRS::new(Some(PARAMETERS))?;
        assert_eq!(
            fsrs.memory_state(item, None).unwrap(),
            MemoryState {
                stability: 31.722975,
                difficulty: 7.382128
            }
        );

        assert_eq!(
            fsrs.next_states(
                Some(MemoryState {
                    stability: 20.925528,
                    difficulty: 7.005062
                }),
                0.9,
                21
            )
            .unwrap()
            .good
            .memory,
            MemoryState {
                stability: 40.874535,
                difficulty: 6.9913807
            }
        );
        Ok(())
    }

    #[test]
    fn test_next_interval() {
        let desired_retentions = (1..=10).map(|i| i as f32 / 10.0).collect::<Vec<_>>();
        let intervals = desired_retentions
            .iter()
            .map(|r| next_interval(1.0, *r).round().max(1.0) as i32)
            .collect::<Vec<_>>();
        assert_eq!(intervals, [422, 102, 43, 22, 13, 8, 4, 2, 1, 1]);
    }

    #[test]
    fn test_evaluate() -> Result<()> {
        let items = anki21_sample_file_converted_to_fsrs();
        let (mut pretrainset, mut trainset): (Vec<FSRSItem>, Vec<FSRSItem>) = items
            .into_iter()
            .partition(|item| item.long_term_review_cnt() == 1);
        (pretrainset, trainset) = filter_outlier(pretrainset, trainset);
        let items = [pretrainset, trainset].concat();

        let fsrs = FSRS::new(Some(&[
            0.669, 1.679, 4.1355, 9.862, 7.9435, 0.9379, 1.0148, 0.1588, 1.3851, 0.1248, 0.8421,
            1.992, 0.153, 0.284, 2.4282, 0.2547, 3.1847, 0.2196, 0.1906,
        ]))?;
        let metrics = fsrs.evaluate(items.clone(), |_| true).unwrap();

        assert_approx_eq([metrics.log_loss, metrics.rmse_bins], [0.208547, 0.035522]);

        let fsrs = FSRS::new(Some(&[]))?;
        let metrics = fsrs.evaluate(items.clone(), |_| true).unwrap();

        assert_approx_eq([metrics.log_loss, metrics.rmse_bins], [0.216326, 0.038727]);

        let fsrs = FSRS::new(Some(PARAMETERS))?;
        let metrics = fsrs.evaluate(items.clone(), |_| true).unwrap();

        assert_approx_eq([metrics.log_loss, metrics.rmse_bins], [0.203049, 0.027558]);

        let (self_by_other, other_by_self) = fsrs
            .universal_metrics(items.clone(), &DEFAULT_PARAMETERS, |_| true)
            .unwrap();

        assert_approx_eq([self_by_other, other_by_self], [0.016236, 0.031085]);

        Ok(())
    }

    #[test]
    fn next_states() -> Result<()> {
        let item = FSRSItem {
            reviews: vec![
                FSRSReview {
                    rating: 1,
                    delta_t: 0,
                },
                FSRSReview {
                    rating: 3,
                    delta_t: 1,
                },
                FSRSReview {
                    rating: 3,
                    delta_t: 3,
                },
                FSRSReview {
                    rating: 3,
                    delta_t: 8,
                },
            ],
        };
        let fsrs = FSRS::new(Some(PARAMETERS))?;
        let state = fsrs.memory_state(item, None).unwrap();
        assert_eq!(
            fsrs.next_states(Some(state), 0.9, 21).unwrap(),
            NextStates {
                again: ItemState {
                    memory: MemoryState {
                        stability: 2.969144,
                        difficulty: 8.000659
                    },
                    interval: 2.9691453
                },
                hard: ItemState {
                    memory: MemoryState {
                        stability: 17.091442,
                        difficulty: 7.6913934
                    },
                    interval: 17.09145
                },
                good: ItemState {
                    memory: MemoryState {
                        stability: 31.722975,
                        difficulty: 7.382128
                    },
                    interval: 31.722988
                },
                easy: ItemState {
                    memory: MemoryState {
                        stability: 71.75015,
                        difficulty: 7.0728626
                    },
                    interval: 71.75018
                }
            }
        );
        assert_eq!(fsrs.next_interval(Some(121.01552), 0.9, 1), 121.01557);
        Ok(())
    }

    #[test]
    fn current_retrievability() {
        let fsrs = FSRS::new(None).unwrap();
        let state = MemoryState {
            stability: 1.0,
            difficulty: 5.0,
        };
        assert_eq!(fsrs.current_retrievability(state, 0), 1.0);
        assert_eq!(fsrs.current_retrievability(state, 1), 0.9);
        assert_eq!(fsrs.current_retrievability(state, 2), 0.82502866);
        assert_eq!(fsrs.current_retrievability(state, 3), 0.76613088);
    }

    #[test]
    fn memory_from_sm2() -> Result<()> {
        let fsrs = FSRS::new(Some(&[]))?;
        let memory_state = fsrs.memory_state_from_sm2(2.5, 10.0, 0.9).unwrap();
        assert_approx_eq(
            [memory_state.stability, memory_state.difficulty],
            [9.999996, 7.079161],
        );
        let memory_state = fsrs.memory_state_from_sm2(2.5, 10.0, 0.8).unwrap();
        assert_approx_eq(
            [memory_state.stability, memory_state.difficulty],
            [4.170096, 9.323614],
        );
        let memory_state = fsrs.memory_state_from_sm2(2.5, 10.0, 0.95).unwrap();
        assert_approx_eq(
            [memory_state.stability, memory_state.difficulty],
            [21.712555, 2.174237],
        );
        let memory_state = fsrs.memory_state_from_sm2(1.3, 20.0, 0.9).unwrap();
        assert_approx_eq(
            [memory_state.stability, memory_state.difficulty],
            [19.999992, 10.0],
        );
        let interval = 15;
        let ease_factor = 2.0;
        let fsrs_factor = fsrs
            .next_states(
                Some(
                    fsrs.memory_state_from_sm2(ease_factor, interval as f32, 0.9)
                        .unwrap(),
                ),
                0.9,
                interval,
            )?
            .good
            .memory
            .stability
            / interval as f32;
        assert!((fsrs_factor - ease_factor).abs() < 0.01);
        Ok(())
    }
}
