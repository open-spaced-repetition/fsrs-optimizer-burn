use crate::batch_shuffle::BatchShuffledDataset;
use crate::cosine_annealing::CosineAnnealingLR;
use crate::dataset::{FSRSBatch, FSRSBatcher, FSRSDataset};
use crate::model::{Model, ModelConfig};
use crate::weight_clipper::weight_clipper;
use burn::module::Module;
use burn::optim::AdamConfig;
use burn::record::{FullPrecisionSettings, PrettyJsonFileRecorder, Recorder};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use burn::train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep};
use burn::{
    config::Config, data::dataloader::DataLoaderBuilder, module::Param, tensor::backend::ADBackend,
    train::LearnerBuilder,
};
use log::info;
use std::path::Path;

impl<B: Backend<FloatElem = f32>> Model<B> {
    fn bceloss(&self, retentions: Tensor<B, 2>, labels: Tensor<B, 2>) -> Tensor<B, 1> {
        let loss: Tensor<B, 2> =
            labels.clone() * retentions.clone().log() + (-labels + 1) * (-retentions + 1).log();
        info!("loss: {}", &loss);
        loss.mean().neg()
    }

    pub fn forward_classification(
        &self,
        t_historys: Tensor<B, 2>,
        r_historys: Tensor<B, 2>,
        delta_ts: Tensor<B, 1>,
        labels: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        // info!("t_historys: {}", &t_historys);
        // info!("r_historys: {}", &r_historys);
        let (stability, _difficulty) = self.forward(t_historys, r_historys);
        let retention = self.power_forgetting_curve(
            delta_ts.clone().unsqueeze::<2>().transpose(),
            stability.clone(),
        );
        let logits = Tensor::cat(vec![-retention.clone() + 1, retention.clone()], 1);
        info!("stability: {}", &stability);
        info!(
            "delta_ts: {}",
            &delta_ts.clone().unsqueeze::<2>().transpose()
        );
        info!("retention: {}", &retention);
        info!("logits: {}", &logits);
        info!(
            "labels: {}",
            &labels.clone().unsqueeze::<2>().float().transpose()
        );
        let loss = self.bceloss(
            retention,
            labels.clone().unsqueeze::<2>().float().transpose(),
        );
        info!("loss: {}", &loss);
        ClassificationOutput::new(loss, logits, labels)
    }
}

impl<B: ADBackend<FloatElem = f32>> Model<B> {
    fn freeze_initial_stability(&self, mut grad: B::Gradients) -> B::Gradients {
        let grad_tensor = self.w.grad(&grad).unwrap();
        let updated_grad_tensor = grad_tensor.slice_assign([0..4], Tensor::zeros([4]));

        self.w.grad_remove(&mut grad);
        self.w.grad_replace(&mut grad, updated_grad_tensor);
        grad
    }
}

impl<B: ADBackend<FloatElem = f32>> TrainStep<FSRSBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: FSRSBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(
            batch.t_historys,
            batch.r_historys,
            batch.delta_ts,
            batch.labels,
        );
        let mut gradients = item.loss.backward();

        if self.freeze_stability {
            gradients = self.freeze_initial_stability(gradients);
        }

        TrainOutput::new(self, gradients, item)
    }

    fn optimize<B1, O>(self, optim: &mut O, lr: f64, grads: burn::optim::GradientsParams) -> Self
    where
        B: ADBackend,
        O: burn::optim::Optimizer<Self, B1>,
        B1: burn::tensor::backend::ADBackend,
        Self: burn::module::ADModule<B1>,
    {
        let mut model = optim.step(lr, self, grads);
        model.w = Param::from(weight_clipper(model.w.val()));
        model
    }
}

impl<B: Backend<FloatElem = f32>> ValidStep<FSRSBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: FSRSBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(
            batch.t_historys,
            batch.r_historys,
            batch.delta_ts,
            batch.labels,
        )
    }
}

static ARTIFACT_DIR: &str = "./tmp/fsrs";

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 512)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 8.0e-3)]
    pub learning_rate: f64,
}

pub fn train<B: ADBackend<FloatElem = f32>>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
) {
    std::fs::create_dir_all(artifact_dir).ok();
    config
        .save(
            Path::new(artifact_dir)
                .join("config.json")
                .to_str()
                .unwrap(),
        )
        .expect("Save without error");

    B::seed(config.seed);

    // Training data
    let dataset = FSRSDataset::sample_dataset();
    let dataset_size = dataset.len();
    let batcher_train = FSRSBatcher::<B>::new(device.clone());
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .build(BatchShuffledDataset::with_seed(
            dataset,
            config.batch_size,
            config.seed,
        ));

    // We don't use any validation data
    let batcher_valid = FSRSBatcher::<B::InnerBackend>::new(device.clone());
    let dataloader_test = DataLoaderBuilder::new(batcher_valid).build(FSRSDataset::from(vec![]));

    let lr_scheduler = CosineAnnealingLR::init(
        (dataset_size * config.num_epochs) as f64,
        config.learning_rate,
    );

    let learner = LearnerBuilder::new(artifact_dir)
        // .metric_train_plot(AccuracyMetric::new())
        // .metric_valid_plot(AccuracyMetric::new())
        // .metric_train_plot(LossMetric::new())
        // .metric_valid_plot(LossMetric::new())
        .with_file_checkpointer(10, PrettyJsonFileRecorder::<FullPrecisionSettings>::new())
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .build(
            config.model.init::<B>(),
            config.optimizer.init(),
            lr_scheduler,
        );

    let mut model_trained = learner.fit(dataloader_train, dataloader_test);
    info!("trained weights: {}", &model_trained.w.val());
    model_trained.w = Param::from(weight_clipper(model_trained.w.val()));
    info!("clipped weights: {}", &model_trained.w.val());

    config
        .save(
            Path::new(ARTIFACT_DIR)
                .join("config.json")
                .to_str()
                .unwrap(),
        )
        .unwrap();

    PrettyJsonFileRecorder::<FullPrecisionSettings>::new()
        .record(
            model_trained.into_record(),
            Path::new(ARTIFACT_DIR).join("model"),
        )
        .expect("Failed to save trained model");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training() {
        if std::env::var("SKIP_TRAINING").is_ok() {
            println!("Skipping test in CI");
            return;
        }
        use burn_ndarray::NdArrayBackend;
        use burn_ndarray::NdArrayDevice;
        type Backend = NdArrayBackend<f32>;
        type AutodiffBackend = burn_autodiff::ADBackendDecorator<Backend>;
        let device = NdArrayDevice::Cpu;

        let artifact_dir = ARTIFACT_DIR;
        train::<AutodiffBackend>(
            artifact_dir,
            TrainingConfig::new(
                ModelConfig {
                    freeze_stability: true,
                },
                AdamConfig::new(),
            ),
            device,
        );
    }
}
