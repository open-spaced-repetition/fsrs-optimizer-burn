use burn::{lr_scheduler::LrScheduler, tensor::backend::Backend, LearningRate};
#[derive(Clone, Debug)]
pub(crate) struct CosineAnnealingLR {
    t_max: f64,
    eta_min: f64,
    init_lr: LearningRate,
    step_count: f64,
    current_lr: LearningRate,
}

impl CosineAnnealingLR {
    pub const fn init(t_max: f64, init_lr: LearningRate) -> Self {
        Self {
            t_max,
            eta_min: 0.0,
            init_lr,
            step_count: 0.0,
            current_lr: init_lr,
        }
    }
}

impl LrScheduler for CosineAnnealingLR {
    type Record<B: Backend> = usize;

    fn step(&mut self) -> LearningRate {
        self.step_count += 1.0;
        use std::f64::consts::PI;
        fn cosine_annealing_lr(
            init_lr: LearningRate,
            lr: LearningRate,
            step_count: f64,
            t_max: f64,
            eta_min: f64,
        ) -> LearningRate {
            let cosine_arg = PI * step_count / t_max;
            if (step_count - 1.0 - t_max) % (2.0 * t_max) == 0.0 {
                (init_lr - eta_min) * (1.0 - f64::cos(PI / t_max)) / 2.0
            } else {
                ((1.0 + f64::cos(cosine_arg)) / (1.0 + f64::cos(PI * (step_count - 1.0) / t_max)))
                    .mul_add(lr - eta_min, eta_min)
            }
        }
        self.current_lr = cosine_annealing_lr(
            self.init_lr,
            self.current_lr,
            self.step_count,
            self.t_max,
            self.eta_min,
        );
        // info!("lr: {}", self.current_lr);
        self.current_lr
    }

    fn to_record<B: Backend>(&self) -> Self::Record<B> {
        self.step_count as usize
    }

    fn load_record<B: Backend>(mut self, record: Self::Record<B>) -> Self {
        self.step_count = record as LearningRate;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    #[test]
    fn lr_scheduler() {
        let mut lr_scheduler = CosineAnnealingLR::init(100000.0, 1.0e-1);

        let lrs = (0..=200000)
            .map(|_| {
                LrScheduler::step(&mut lr_scheduler);
                lr_scheduler.current_lr
            })
            .step_by(20000)
            .collect::<Vec<_>>();

        TensorData::from(&lrs[..]).assert_approx_eq(
            &TensorData::from([
                0.1,
                0.09045084971874785,
                0.06545084971874875,
                0.034549150281253875,
                0.009549150281252989,
                0.0,
                0.009549150281252692,
                0.03454915028125239,
                0.06545084971874746,
                0.09045084971874952,
                0.10000000000000353,
            ]),
            5,
        );
    }
}
