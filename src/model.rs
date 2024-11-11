use crate::error::{FSRSError, Result};
use crate::inference::{Parameters, DECAY, D_MAX, D_MIN, FACTOR, S_MAX, S_MIN};
use crate::parameter_clipper::clip_parameters;
use crate::DEFAULT_PARAMETERS;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use burn::{
    config::Config,
    module::{Module, Param},
    tensor::{backend::Backend, Data, Shape, Tensor},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub w: Param<Tensor<B, 1>>,
    pub config: ModelConfig,
}

pub(crate) trait Get<B: Backend, const N: usize> {
    fn get(&self, n: usize) -> Tensor<B, N>;
}

impl<B: Backend, const N: usize> Get<B, N> for Tensor<B, N> {
    fn get(&self, n: usize) -> Self {
        self.clone().slice([n..(n + 1)])
    }
}

trait Pow<B: Backend, const N: usize> {
    // https://github.com/burn-rs/burn/issues/590 , after that finished, just remove this trait and below impl, all will ok.
    fn pow(&self, other: Tensor<B, N>) -> Tensor<B, N>;
}

impl<B: Backend, const N: usize> Pow<B, N> for Tensor<B, N> {
    fn pow(&self, other: Self) -> Self {
        // a ^ b => exp(ln(a^b)) => exp(b ln (a))
        (self.clone().log() * other).exp()
    }
}

impl<B: Backend> Model<B> {
    #[allow(clippy::new_without_default)]
    pub fn new(config: ModelConfig) -> Self {
        let initial_params = config
            .initial_stability
            .unwrap_or_else(|| DEFAULT_PARAMETERS[0..4].try_into().unwrap())
            .into_iter()
            .chain(DEFAULT_PARAMETERS[4..].iter().copied())
            .collect();

        Self {
            w: Param::from_tensor(Tensor::from_floats(
                Data::new(initial_params, Shape { dims: [19] }),
                &B::Device::default(),
            )),
            config,
        }
    }

    pub fn power_forgetting_curve(&self, t: Tensor<B, 1>, s: Tensor<B, 1>) -> Tensor<B, 1> {
        (t / s * FACTOR + 1).powf_scalar(DECAY as f32)
    }

    fn stability_after_success(
        &self,
        last_s: Tensor<B, 1>,
        last_d: Tensor<B, 1>,
        r: Tensor<B, 1>,
        rating: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let batch_size = rating.dims()[0];
        let hard_penalty = Tensor::ones([batch_size], &B::Device::default())
            .mask_where(rating.clone().equal_elem(2), self.w.get(15));
        let easy_bonus = Tensor::ones([batch_size], &B::Device::default())
            .mask_where(rating.equal_elem(4), self.w.get(16));

        last_s.clone()
            * (self.w.get(8).exp()
                * (-last_d + 11)
                * (last_s.pow(-self.w.get(9)))
                * (((-r + 1) * self.w.get(10)).exp() - 1)
                * hard_penalty
                * easy_bonus
                + 1)
    }

    fn stability_after_failure(
        &self,
        last_s: Tensor<B, 1>,
        last_d: Tensor<B, 1>,
        r: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let new_s = self.w.get(11)
            * last_d.pow(-self.w.get(12))
            * ((last_s.clone() + 1).pow(self.w.get(13)) - 1)
            * ((-r + 1) * self.w.get(14)).exp();
        new_s
            .clone()
            .mask_where(last_s.clone().lower(new_s), last_s)
    }

    fn stability_short_term(&self, last_s: Tensor<B, 1>, rating: Tensor<B, 1>) -> Tensor<B, 1> {
        (last_s * (self.w.get(17) * (rating - 3 + self.w.get(18))).exp()).clamp(S_MIN, S_MAX)
    }

    fn mean_reversion(&self, new_d: Tensor<B, 1>) -> Tensor<B, 1> {
        let rating = Tensor::from_floats([4.0], &B::Device::default());
        (self.w.get(7) * (self.init_difficulty(rating) - new_d.clone()) + new_d).clamp(D_MIN, D_MAX)
    }

    pub(crate) fn init_stability(&self, rating: Tensor<B, 1>) -> Tensor<B, 1> {
        (self.w.val().select(0, rating.int() - 1)).clamp(S_MIN, S_MAX)
    }

    fn init_difficulty(&self, rating: Tensor<B, 1>) -> Tensor<B, 1> {
        (self.w.get(4) - (self.w.get(5) * (rating - 1)).exp() + 1).clamp(D_MIN, D_MAX)
    }

    fn linear_damping(&self, delta_d: Tensor<B, 1>, old_d: Tensor<B, 1>) -> Tensor<B, 1> {
        old_d.neg().add_scalar(10.0) * delta_d.div_scalar(9.0)
    }

    fn next_difficulty(&self, difficulty: Tensor<B, 1>, rating: Tensor<B, 1>) -> Tensor<B, 1> {
        let delta_d = -self.w.get(6) * (rating - 3);
        (difficulty.clone() + self.linear_damping(delta_d, difficulty)).clamp(D_MIN, D_MAX)
    }

    pub(crate) fn step(
        &self,
        delta_t: Tensor<B, 1>,
        rating: Tensor<B, 1>,
        state: Option<MemoryStateTensors<B>>,
    ) -> MemoryStateTensors<B> {
        let (new_s, new_d) = if let Some(state) = state {
            let retention = self.power_forgetting_curve(delta_t.clone(), state.stability.clone());
            let stability_after_success = self.stability_after_success(
                state.stability.clone(),
                state.difficulty.clone(),
                retention.clone(),
                rating.clone(),
            );
            let stability_after_failure = self.stability_after_failure(
                state.stability.clone(),
                state.difficulty.clone(),
                retention,
            );
            let stability_short_term =
                self.stability_short_term(state.stability.clone(), rating.clone());
            let mut new_stability = stability_after_success
                .mask_where(rating.clone().equal_elem(1), stability_after_failure);
            new_stability = new_stability.mask_where(delta_t.equal_elem(0), stability_short_term);

            let mut new_difficulty = self.next_difficulty(state.difficulty.clone(), rating.clone());
            new_difficulty = self.mean_reversion(new_difficulty);
            // mask padding zeros for rating
            new_stability = new_stability.mask_where(rating.clone().equal_elem(0), state.stability);
            new_difficulty = new_difficulty.mask_where(rating.equal_elem(0), state.difficulty);
            (new_stability, new_difficulty)
        } else {
            (
                self.init_stability(rating.clone()),
                self.init_difficulty(rating),
            )
        };
        MemoryStateTensors {
            stability: new_s.clamp(S_MIN, S_MAX),
            difficulty: new_d,
        }
    }

    /// If [starting_state] is provided, it will be used instead of the default initial stability/
    /// difficulty.
    pub(crate) fn forward(
        &self,
        delta_ts: Tensor<B, 2>,
        ratings: Tensor<B, 2>,
        starting_state: Option<MemoryStateTensors<B>>,
    ) -> MemoryStateTensors<B> {
        let [seq_len, _batch_size] = delta_ts.dims();
        let mut state = starting_state;
        for i in 0..seq_len {
            let delta_t = delta_ts.get(i).squeeze(0);
            // [batch_size]
            let rating = ratings.get(i).squeeze(0);
            // [batch_size]
            state = Some(self.step(delta_t, rating, state));
        }
        state.unwrap()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MemoryStateTensors<B: Backend> {
    pub stability: Tensor<B, 1>,
    pub difficulty: Tensor<B, 1>,
}

#[derive(Config, Module, Debug, Default)]
pub struct ModelConfig {
    #[config(default = false)]
    pub freeze_stability: bool,
    pub initial_stability: Option<[f32; 4]>,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self) -> Model<B> {
        Model::new(self.clone())
    }
}

/// This is the main structure provided by this crate. It can be used
/// for both parameter training, and for reviews.
#[derive(Debug, Clone)]
pub struct FSRS<B: Backend = NdArray> {
    model: Option<Model<B>>,
    device: B::Device,
}

impl FSRS<NdArray> {
    /// - Parameters must be provided before running commands that need them.
    /// - Parameters may be an empty slice to use the default values instead.
    pub fn new(parameters: Option<&Parameters>) -> Result<Self> {
        Self::new_with_backend(parameters, NdArrayDevice::Cpu)
    }
}

impl<B: Backend> FSRS<B> {
    pub fn new_with_backend<B2: Backend>(
        parameters: Option<&Parameters>,
        device: B2::Device,
    ) -> Result<FSRS<B2>> {
        let model = match parameters {
            Some(params) => {
                let parameters = check_and_fill_parameters(params)?;
                let model = parameters_to_model::<B2>(&parameters);
                Some(model)
            }
            None => None,
        };

        Ok(FSRS { model, device })
    }

    pub(crate) fn model(&self) -> &Model<B> {
        self.model
            .as_ref()
            .expect("command requires parameters to be set on creation")
    }

    pub(crate) fn device(&self) -> B::Device {
        self.device.clone()
    }
}

pub(crate) fn parameters_to_model<B: Backend>(parameters: &Parameters) -> Model<B> {
    let config = ModelConfig::default();
    let mut model = Model::new(config);
    model.w = Param::from_tensor(Tensor::from_floats(
        Data::new(clip_parameters(parameters), Shape { dims: [19] }),
        &B::Device::default(),
    ));
    model
}

pub(crate) fn check_and_fill_parameters(parameters: &Parameters) -> Result<Vec<f32>, FSRSError> {
    let parameters = match parameters.len() {
        0 => DEFAULT_PARAMETERS.to_vec(),
        17 => {
            let mut parameters = parameters.to_vec();
            parameters[4] = parameters[5].mul_add(2.0, parameters[4]);
            parameters[5] = parameters[5].mul_add(3.0, 1.0).ln() / 3.0;
            parameters[6] += 0.5;
            parameters.extend_from_slice(&[0.0, 0.0]);
            parameters
        }
        19 => parameters.to_vec(),
        _ => return Err(FSRSError::InvalidParameters),
    };
    if parameters.iter().any(|&w| !w.is_finite()) {
        return Err(FSRSError::InvalidParameters);
    }
    Ok(parameters)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{Model, Tensor};
    use burn::tensor::Data;

    #[test]
    fn w() {
        let model = Model::new(ModelConfig::default());
        assert_eq!(model.w.val().to_data(), Data::from(DEFAULT_PARAMETERS))
    }

    #[test]
    fn convert_parameters() {
        let fsrs4dot5_param = vec![
            0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 1.26,
            0.29, 2.61,
        ];
        let fsrs5_param = check_and_fill_parameters(&fsrs4dot5_param).unwrap();
        assert_eq!(
            fsrs5_param,
            vec![
                0.4, 0.6, 2.4, 5.8, 6.81, 0.44675013, 1.36, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05,
                0.34, 1.26, 0.29, 2.61, 0.0, 0.0,
            ]
        )
    }

    #[test]
    fn power_forgetting_curve() {
        let device = NdArrayDevice::Cpu;
        let model = Model::new(ModelConfig::default());
        let delta_t = Tensor::from_floats([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &device);
        let stability = Tensor::from_floats([1.0, 2.0, 3.0, 4.0, 4.0, 2.0], &device);
        let retention = model.power_forgetting_curve(delta_t, stability);
        assert_eq!(
            retention.to_data(),
            Data::from([1.0, 0.946059, 0.9299294, 0.9221679, 0.90000004, 0.79394597])
        )
    }

    #[test]
    fn init_stability() {
        let device = NdArrayDevice::Cpu;
        let model = Model::new(ModelConfig::default());
        let rating = Tensor::from_floats([1.0, 2.0, 3.0, 4.0, 1.0, 2.0], &device);
        let stability = model.init_stability(rating);
        assert_eq!(
            stability.to_data(),
            Data::from([
                DEFAULT_PARAMETERS[0],
                DEFAULT_PARAMETERS[1],
                DEFAULT_PARAMETERS[2],
                DEFAULT_PARAMETERS[3],
                DEFAULT_PARAMETERS[0],
                DEFAULT_PARAMETERS[1]
            ])
        )
    }

    #[test]
    fn init_difficulty() {
        let device = NdArrayDevice::Cpu;
        let model = Model::new(ModelConfig::default());
        let rating = Tensor::from_floats([1.0, 2.0, 3.0, 4.0, 1.0, 2.0], &device);
        let difficulty = model.init_difficulty(rating);
        assert_eq!(
            difficulty.to_data(),
            Data::from([
                DEFAULT_PARAMETERS[4],
                DEFAULT_PARAMETERS[4] - DEFAULT_PARAMETERS[5].exp() + 1.0,
                DEFAULT_PARAMETERS[4] - (2.0 * DEFAULT_PARAMETERS[5]).exp() + 1.0,
                DEFAULT_PARAMETERS[4] - (3.0 * DEFAULT_PARAMETERS[5]).exp() + 1.0,
                DEFAULT_PARAMETERS[4],
                DEFAULT_PARAMETERS[4] - DEFAULT_PARAMETERS[5].exp() + 1.0,
            ])
        )
    }

    #[test]
    fn forward() {
        let device = NdArrayDevice::Cpu;
        let model = Model::new(ModelConfig::default());
        let delta_ts = Tensor::from_floats(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
            ],
            &device,
        );
        let ratings = Tensor::from_floats(
            [
                [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
                [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
            ],
            &device,
        );
        let state = model.forward(delta_ts, ratings, None);
        dbg!(&state);
    }

    #[test]
    fn next_difficulty() {
        let device = NdArrayDevice::Cpu;
        let model = Model::new(ModelConfig::default());
        let difficulty = Tensor::from_floats([5.0; 4], &device);
        let rating = Tensor::from_floats([1.0, 2.0, 3.0, 4.0], &device);
        let next_difficulty = model.next_difficulty(difficulty, rating);
        next_difficulty.clone().backward();
        assert_eq!(
            next_difficulty.to_data(),
            Data::from([6.622667, 5.811333, 5.0, 4.188667])
        );
        let next_difficulty = model.mean_reversion(next_difficulty);
        next_difficulty.clone().backward();
        assert_eq!(
            next_difficulty.to_data(),
            Data::from([6.607035, 5.7994337, 4.9918327, 4.1842318])
        )
    }

    #[test]
    fn next_stability() {
        let device = NdArrayDevice::Cpu;
        let model = Model::new(ModelConfig::default());
        let stability = Tensor::from_floats([5.0; 4], &device);
        let difficulty = Tensor::from_floats([1.0, 2.0, 3.0, 4.0], &device);
        let retention = Tensor::from_floats([0.9, 0.8, 0.7, 0.6], &device);
        let rating = Tensor::from_floats([1.0, 2.0, 3.0, 4.0], &device);
        let s_recall = model.stability_after_success(
            stability.clone(),
            difficulty.clone(),
            retention.clone(),
            rating.clone(),
        );
        s_recall.clone().backward();
        assert_eq!(
            s_recall.to_data(),
            Data::from([25.77614, 14.121894, 60.40441, 208.97597])
        );
        let s_forget = model.stability_after_failure(stability.clone(), difficulty, retention);
        s_forget.clone().backward();
        assert_eq!(
            s_forget.to_data(),
            Data::from([1.7028502, 1.9798818, 2.3759942, 2.8885393])
        );
        let next_stability = s_recall.mask_where(rating.clone().equal_elem(1), s_forget);
        next_stability.clone().backward();
        assert_eq!(
            next_stability.to_data(),
            Data::from([1.7028502, 14.121894, 60.40441, 208.97597])
        );
        let next_stability = model.stability_short_term(stability, rating);
        assert_eq!(
            next_stability.to_data(),
            Data::from([2.5051427, 4.199207, 7.038856, 11.798775])
        )
    }

    #[test]
    fn fsrs() {
        assert!(FSRS::new(Some(&[])).is_ok());
        assert!(FSRS::new(Some(&[1.])).is_err());
        assert!(FSRS::new(Some(DEFAULT_PARAMETERS.as_slice())).is_ok());
        assert!(FSRS::new(Some(&DEFAULT_PARAMETERS[..17])).is_ok());
    }
}
