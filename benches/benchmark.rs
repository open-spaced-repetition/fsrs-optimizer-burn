// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

use std::hint::black_box;
use std::iter::repeat;

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use fsrs::FSRS;
use fsrs::FSRSReview;
use fsrs::NextStates;
use fsrs::SimulatorConfig;
use fsrs::{FSRSItem, MemoryState};
use itertools::Itertools;

pub(crate) fn calc_mem(inf: &FSRS, past_reviews: usize) -> MemoryState {
    let review = FSRSReview {
        rating: 3,
        delta_t: 21,
    };
    let reviews = repeat(review).take(past_reviews + 1).collect_vec();
    inf.memory_state(FSRSItem { reviews }, None).unwrap()
}

pub(crate) fn next_states(inf: &FSRS) -> NextStates {
    inf.next_states(
        Some(MemoryState {
            stability: 51.344814,
            difficulty: 7.005062,
        }),
        0.9,
        21,
    )
    .unwrap()
}

pub(crate) fn optimal_retention(inf: &FSRS, config: &SimulatorConfig) -> f32 {
    inf.optimal_retention(config, &[], |_v| true).unwrap()
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let fsrs = FSRS::new(Some(&[
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
    ]))
    .unwrap();
    let config = SimulatorConfig {
        deck_size: 3650,
        learn_span: 365,
        max_cost_perday: f32::INFINITY,
        learn_limit: 10,
        loss_aversion: 1.0,
        ..Default::default()
    };
    c.bench_function("calc_mem", |b| b.iter(|| black_box(calc_mem(&fsrs, 100))));
    c.bench_function("next_states", |b| b.iter(|| black_box(next_states(&fsrs))));
    c.bench_function("optimal_retention", |b| {
        b.iter(|| black_box(optimal_retention(&fsrs, &config)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
