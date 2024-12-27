use std::sync::Mutex;

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::{DataLoaderIterator, Progress};
use burn::prelude::Backend;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::dataset::{FSRSBatch, FSRSBatcher, FSRSDataset};

#[derive(Clone)]
pub(crate) struct BatchTensorDataset<B: Backend> {
    dataset: Vec<FSRSBatch<B>>,
}

impl<B: Backend> BatchTensorDataset<B> {
    /// Creates a new shuffled dataset.
    pub fn new(dataset: FSRSDataset, batch_size: usize, device: B::Device) -> Self {
        let batcher = FSRSBatcher::<B>::new(device);
        let dataset = dataset
            .items
            .chunks(batch_size)
            .map(|items| batcher.batch(items.to_vec()))
            .collect();
        Self { dataset }
    }
}

impl<B: Backend> BatchTensorDataset<B> {
    fn get(&self, index: usize) -> Option<FSRSBatch<B>> {
        self.dataset.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

pub struct ShuffleDataLoader<B: Backend> {
    dataset: BatchTensorDataset<B>,
    rng: Mutex<rand::rngs::StdRng>,
}

impl<B: Backend> ShuffleDataLoader<B> {
    pub fn new(dataset: BatchTensorDataset<B>, seed: u64) -> Self {
        Self {
            dataset,
            rng: Mutex::new(rand::rngs::StdRng::seed_from_u64(seed)),
        }
    }
}

pub(crate) struct ShuffleDataLoaderIterator<B: Backend> {
    current_index: usize,
    indices: Vec<usize>,
    dataset: BatchTensorDataset<B>,
}

impl<B: Backend> ShuffleDataLoaderIterator<B> {
    pub(crate) fn new(dataset: BatchTensorDataset<B>, indices: Vec<usize>) -> Self {
        Self {
            current_index: 0,
            indices,
            dataset,
        }
    }

    pub(crate) fn progress(&self) -> Progress {
        Progress {
            items_processed: self.current_index,
            items_total: self.dataset.len(),
        }
    }
}

impl<B: Backend> Iterator for ShuffleDataLoaderIterator<B> {
    type Item = FSRSBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(index) = self.indices.get(self.current_index) {
            self.current_index += 1;
            return self.dataset.get(*index);
        }
        None
    }
}

impl<B: Backend> DataLoaderIterator<FSRSBatch<B>> for ShuffleDataLoaderIterator<B> {
    fn progress(&self) -> Progress {
        Progress::new(self.current_index, self.dataset.len())
    }
}

impl<B: Backend> ShuffleDataLoader<B> {
    pub(crate) fn iter(&self) -> ShuffleDataLoaderIterator<B> {
        let mut indices: Vec<_> = (0..self.dataset.len()).collect();
        indices.shuffle(&mut *self.rng.lock().unwrap());
        ShuffleDataLoaderIterator::new(self.dataset.clone(), indices)
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::{ndarray::NdArrayDevice, NdArray},
        tensor::Shape,
    };

    use super::*;
    use crate::{
        convertor_tests::anki21_sample_file_converted_to_fsrs, dataset::prepare_training_data,
    };

    #[test]
    fn test_simple_dataloader() {
        let train_set = anki21_sample_file_converted_to_fsrs();
        let (_pre_train_set, train_set) = prepare_training_data(train_set);
        let dataset = FSRSDataset::from(train_set);
        let batch_size = 512;
        let seed = 114514;
        let device = NdArrayDevice::Cpu;
        type Backend = NdArray<f32>;

        let dataset = BatchTensorDataset::<Backend>::new(dataset, batch_size, device);
        let dataloader = ShuffleDataLoader::new(dataset, seed);
        let mut iterator = dataloader.iter();
        // dbg!(&iterator.indices);
        let batch = iterator.next().unwrap();
        assert_eq!(
            batch.t_historys.shape(),
            Shape {
                dims: vec![7, batch_size]
            }
        );
        let batch = iterator.next().unwrap();
        assert_eq!(
            batch.t_historys.shape(),
            Shape {
                dims: vec![6, batch_size]
            }
        );

        let lengths = iterator
            .map(|batch| batch.t_historys.shape().dims[0])
            .collect::<Vec<_>>();
        assert_eq!(
            lengths,
            vec![
                48, 6, 8, 5, 11, 5, 10, 19, 6, 13, 9, 6, 5, 3, 9, 6, 3, 13, 7, 5, 4, 4, 4, 6, 4, 3,
            ]
        );

        let mut iterator = dataloader.iter();
        // dbg!(&iterator.indices);
        let batch = iterator.next().unwrap();
        assert_eq!(
            batch.t_historys.shape(),
            Shape {
                dims: vec![19, batch_size]
            }
        );
        let batch = iterator.next().unwrap();
        assert_eq!(
            batch.t_historys.shape(),
            Shape {
                dims: vec![9, batch_size]
            }
        );

        let lengths = iterator
            .map(|batch| batch.t_historys.shape().dims[0])
            .collect::<Vec<_>>();
        assert_eq!(
            lengths,
            vec![3, 11, 3, 6, 6, 6, 5, 5, 7, 6, 4, 9, 10, 4, 48, 3, 4, 5, 13, 13, 7, 5, 4, 8, 6, 6]
        );
    }
}
