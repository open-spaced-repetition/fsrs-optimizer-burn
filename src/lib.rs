#![deny(warnings)]
#![allow(clippy::single_range_in_vec_init)]
#![allow(clippy::redundant_pub_crate)]
mod batch_shuffle;
#[cfg(test)]
mod convertor_tests;
mod cosine_annealing;
mod dataset;
mod error;
mod inference;
mod model;
mod optimal_retention;
mod parameter_clipper;
mod pre_training;
#[cfg(test)]
mod test_helpers;
mod training;

pub use dataset::{FSRSItem, FSRSReview};
pub use error::{FSRSError, Result};
pub use inference::{
    DEFAULT_PARAMETERS, ItemProgress, ItemState, MemoryState, ModelEvaluation, NextStates,
};
pub use model::FSRS;
pub use optimal_retention::{
    Card, RevlogEntry, RevlogReviewKind, SimulatorConfig, extract_simulator_config, simulate,
};
pub use training::CombinedProgressState;
