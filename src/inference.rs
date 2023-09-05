use std::collections::HashMap;
use std::ops::{Add, Sub};

use crate::model::ModelConfig;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArrayBackend;
use burn::module::Param;
use burn::tensor::{Data, Shape, Tensor};
use burn::{data::dataloader::batcher::Batcher, tensor::backend::Backend};

use crate::dataset::FSRSBatch;
use crate::dataset::FSRSBatcher;
use crate::error::Result;
use crate::model::Model;
use crate::training::BCELoss;
use crate::{FSRSError, FSRSItem};

fn infer<B: Backend<FloatElem = f32>>(
    model: &Model<B>,
    batch: FSRSBatch<B>,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    let (stability, difficulty) = model.forward(batch.t_historys, batch.r_historys);
    let retention = model.power_forgetting_curve(
        batch.delta_ts.clone().unsqueeze::<2>().transpose(),
        stability.clone(),
    );
    (stability, difficulty, retention)
}

#[derive(Debug, Clone, Copy)]
pub struct ItemProgress {
    pub current: usize,
    pub total: usize,
}

pub fn evaluate<F>(weights: [f32; 17], items: Vec<FSRSItem>, mut progress: F) -> Result<(f32, f32)>
where
    F: FnMut(ItemProgress) -> bool,
{
    type Backend = NdArrayBackend<f32>;
    let device = NdArrayDevice::Cpu;
    let batcher = FSRSBatcher::<Backend>::new(device);
    let config = ModelConfig::default();
    let mut model = Model::<Backend>::new(config);
    model.w = Param::from(Tensor::from_floats(Data::new(
        weights.to_vec(),
        Shape { dims: [17] },
    )));
    let mut all_pred = vec![];
    let mut all_true_val = vec![];
    let mut all_retention = vec![];
    let mut all_labels = vec![];
    let mut progress_info = ItemProgress {
        current: 0,
        total: items.len(),
    };
    for chunk in items.chunks(512) {
        let batch = batcher.batch(chunk.to_vec());
        let (_stability, _difficulty, retention) = infer::<Backend>(&model, batch.clone());
        let pred = retention.clone().squeeze::<1>(1).to_data().value;
        all_pred.extend(pred);
        let true_val = batch.labels.clone().float().to_data().value;
        all_true_val.extend(true_val);
        all_retention.push(retention);
        all_labels.push(batch.labels);
        progress_info.current += chunk.len();
        if !progress(progress_info) {
            return Err(FSRSError::Interrupted);
        }
    }
    let rmse = calibration_rmse(all_pred, all_true_val);
    let all_retention = Tensor::cat(all_retention, 0);
    let all_labels = Tensor::cat(all_labels, 0)
        .unsqueeze::<2>()
        .float()
        .transpose();
    let loss = BCELoss::<Backend>::new().forward(all_retention, all_labels);
    Ok((loss.to_data().value[0], rmse))
}

fn get_bin(x: f32, bins: i32) -> i32 {
    let log_base = (bins.add(1) as f32).ln();
    let binned_x = (x * log_base).exp().floor().sub(1.0);
    (binned_x as i32).min(bins - 1).max(0)
}

fn calibration_rmse(pred: Vec<f32>, true_val: Vec<f32>) -> f32 {
    if pred.len() != true_val.len() {
        panic!("Vectors pred and true_val must have the same length");
    }

    let mut groups = HashMap::new();

    for (p, t) in pred.iter().zip(true_val) {
        let bin = get_bin(*p, 20);
        groups.entry(bin).or_insert_with(Vec::new).push((p, t));
    }

    let mut total_sum = 0.0;
    let mut total_count = 0.0;

    for (_bin, group) in groups.iter() {
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
    use crate::convertor_tests::anki21_sample_file_converted_to_fsrs;

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
    fn test_evaluate() {
        let items = anki21_sample_file_converted_to_fsrs();

        let metrics = evaluate(
            [
                0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34,
                1.26, 0.29, 2.61,
            ],
            items.clone(),
            |_| true,
        )
        .unwrap();

        Data::from([metrics.0, metrics.1])
            .assert_approx_eq(&Data::from([0.20820294, 0.042998276]), 5);

        let metrics = evaluate(
            [
                0.81497127,
                1.5411042,
                4.007436,
                9.045982,
                4.9264183,
                1.039322,
                0.93803364,
                0.0,
                1.5530516,
                0.10299722,
                0.9981442,
                2.210701,
                0.018248068,
                0.3422524,
                1.3384504,
                0.22278537,
                2.6646678,
            ],
            items,
            |_| true,
        )
        .unwrap();

        Data::from([metrics.0, metrics.1])
            .assert_approx_eq(&Data::from([0.20206251, 0.017628053]), 5);
    }
}