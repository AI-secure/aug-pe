# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""k-NN precision and recall."""

import numpy as np
from time import time
# example of calculating the frechet inception distance

from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
import os

# calculate inception score with Keras
import torch
from sklearn.metrics import pairwise_distances
import argparse
import csv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datasets import load_dataset


# ----------------------------------------------------------------------------

class DistanceBlock():
    """Provides multi-GPU support to calculate pairwise distances between two batches of feature vectors."""

    def __init__(self, num_features, num_gpus):
        self.num_features = num_features
        self.num_gpus = num_gpus

    def pairwise_distances(self, U, V):
        """Evaluate pairwise distances between two batches of feature vectors."""
        output = pairwise_distances(U, V, n_jobs=24)
        return output


# ----------------------------------------------------------------------------

class ManifoldEstimator():
    """Estimates the manifold of given feature vectors."""

    def __init__(self, distance_block, features, row_batch_size=25000, col_batch_size=50000,
                 nhood_sizes=[3], clamp_to_percentile=None, eps=1e-5):
        """Estimate the manifold of given feature vectors.

            Args:
                distance_block: DistanceBlock object that distributes pairwise distance
                    calculation to multiple GPUs.
                features (np.array/tf.Tensor): Matrix of feature vectors to estimate their manifold.
                row_batch_size (int): Row batch size to compute pairwise distances
                    (parameter to trade-off between memory usage and performance).
                col_batch_size (int): Column batch size to compute pairwise distances.
                nhood_sizes (list): Number of neighbors used to estimate the manifold.
                clamp_to_percentile (float): Prune hyperspheres that have radius larger than
                    the given percentile.
                eps (float): Small number for numerical stability.
        """
        num_images = features.shape[0]
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.eps = eps
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self._ref_features = features
        self._distance_block = distance_block

        # Estimate manifold of features by calculating distances to k-NN of each sample.
        self.D = np.zeros([num_images, self.num_nhoods], dtype=np.float32)
        distance_batch = np.zeros(
            [row_batch_size, num_images], dtype=np.float32)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, num_images, row_batch_size):
            end1 = min(begin1 + row_batch_size, num_images)
            row_batch = features[begin1:end1]

            for begin2 in range(0, num_images, col_batch_size):
                end2 = min(begin2 + col_batch_size, num_images)
                col_batch = features[begin2:end2]

                # Compute distances between batches.
                distance_batch[0:end1 - begin1, begin2:end2] = self._distance_block.pairwise_distances(row_batch,
                                                                                                       col_batch)

            # Find the k-nearest neighbor from the current batch.
            self.D[begin1:end1, :] = np.partition(
                distance_batch[0:end1 - begin1, :], seq, axis=1)[:, self.nhood_sizes]

        if clamp_to_percentile is not None:
            max_distances = np.percentile(self.D, clamp_to_percentile, axis=0)
            self.D[self.D > max_distances] = 0

    def evaluate(self, eval_features, return_realism=False, return_neighbors=False):
        """Evaluate if new feature vectors are at the manifold."""
        num_eval_images = eval_features.shape[0]
        num_ref_images = self.D.shape[0]
        distance_batch = np.zeros(
            [self.row_batch_size, num_ref_images], dtype=np.float32)
        batch_predictions = np.zeros(
            [num_eval_images, self.num_nhoods], dtype=np.int32)
        max_realism_score = np.zeros([num_eval_images, ], dtype=np.float32)
        nearest_indices = np.zeros([num_eval_images, ], dtype=np.int32)

        for begin1 in range(0, num_eval_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval_images)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_images)
                ref_batch = self._ref_features[begin2:end2]

                distance_batch[0:end1 - begin1, begin2:end2] = self._distance_block.pairwise_distances(feature_batch,
                                                                                                       ref_batch)

            # From the minibatch of new feature vectors, determine if they are in the estimated manifold.
            # If a feature vector is inside a hypersphere of some reference sample, then
            # the new sample lies at the estimated manifold.
            # The radii of the hyperspheres are determined from distances of neighborhood size k.
            samples_in_manifold = distance_batch[0:end1 -
                                                 begin1, :, None] <= self.D
            batch_predictions[begin1:end1] = np.any(
                samples_in_manifold, axis=1).astype(np.int32)

            max_realism_score[begin1:end1] = np.max(self.D[:, 0] / (distance_batch[0:end1 - begin1, :] + self.eps),
                                                    axis=1)
            nearest_indices[begin1:end1] = np.argmin(
                distance_batch[0:end1 - begin1, :], axis=1)

        if return_realism and return_neighbors:
            return batch_predictions, max_realism_score, nearest_indices
        elif return_realism:
            return batch_predictions, max_realism_score
        elif return_neighbors:
            return batch_predictions, nearest_indices

        return batch_predictions


# ----------------------------------------------------------------------------

def knn_precision_recall_features(ref_features, eval_features, nhood_sizes=[3],
                                  row_batch_size=10000, col_batch_size=50000, num_gpus=1):
    """Calculates k-NN precision and recall for two sets of feature vectors.

        Args:
            ref_features (np.array/tf.Tensor): Feature vectors of reference images.
            eval_features (np.array/tf.Tensor): Feature vectors of generated images.
            nhood_sizes (list): Number of neighbors used to estimate the manifold.
            row_batch_size (int): Row batch size to compute pairwise distances
                (parameter to trade-off between memory usage and performance).
            col_batch_size (int): Column batch size to compute pairwise distances.
            num_gpus (int): Number of GPUs used to evaluate precision and recall.

        Returns:
            State (dict): Dict that contains precision and recall calculated from
            ref_features and eval_features.
    """
    state = dict()
    num_images = ref_features.shape[0]
    num_features = ref_features.shape[1]

    # Initialize DistanceBlock and ManifoldEstimators.
    distance_block = DistanceBlock(num_features, num_gpus)
    ref_manifold = ManifoldEstimator(
        distance_block, ref_features, row_batch_size, col_batch_size, nhood_sizes)
    eval_manifold = ManifoldEstimator(
        distance_block, eval_features, row_batch_size, col_batch_size, nhood_sizes)

    # Evaluate precision and recall using k-nearest neighbors.
    print('Evaluating k-NN precision and recall with %i samples...' % num_images)
    start = time()

    # Precision: How many points from eval_features are in ref_features manifold.
    precision = ref_manifold.evaluate(eval_features)
    state['precision'] = precision.mean(axis=0).item()

    # Recall: How many points from ref_features are in eval_features manifold.
    recall = eval_manifold.evaluate(ref_features)
    state['recall'] = recall.mean(axis=0).item()
    if state['precision']+state['recall'] > 0:
        state['f1'] = (2*state['precision'] * state['recall']) / \
            (state['precision']+state['recall'])
    else:
        state['f1'] = 0
    print('Evaluated k-NN precision and recall in: %gs' % (time() - start))

    return state


# ----------------------------------------------------------------------------

def balance_dataset(dataset, sample_size=5000):
    training_dataset = dataset['train']
    sample_indices = []
    # for label in ['stars:1.0', 'stars:2.0', 'stars:3.0', 'stars:4.0', 'stars:5.0']:
    for label in ['Review Stars: 1.0', 'Review Stars: 2.0', 'Review Stars: 3.0', 'Review Stars: 4.0', 'Review Stars: 5.0']:
        indices = np.where(np.array(training_dataset['label2']) == label)[0]
        sample_indices.append(np.random.choice(
            indices, size=sample_size, replace=False))
    sample_indices = np.concatenate(sample_indices)
    np.random.shuffle(sample_indices)
    training_dataset = training_dataset.select(sample_indices)
    dataset['train'] = training_dataset
    return dataset


# calculate frechet inception distance
def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def log_embeddings(embeddings, additional_info, folder, fname=''):
    if not os.path.exists(folder):
        os.makedirs(folder)
    savefname = os.path.join(folder, fname+'.embeddings.npz')
    print("save embeddings into", savefname)
    np.savez(
        savefname,
        embeddings=embeddings,
        additional_info=additional_info)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--original_file", type=str,
                        default="yelp_dataset_sample/train.csv", required=False)
    parser.add_argument("--synthetic_file", type=str,
                        default="yelp_dataset_sample/GPT2-eps8-finetuned-prompt-seq64_512.generations.csv",
                        required=False)
    parser.add_argument("--model_name_or_path", type=str,
                        default="all-mpnet-base-v2", required=False)
    parser.add_argument("--batch_size", type=int, required=False, default=128)
    parser.add_argument("--k", type=int, required=False, default=3)

    args = parser.parse_args()

    ori_data = load_dataset("csv", data_files=args.original_file)
    syn_data = load_dataset("csv", data_files=args.synthetic_file)

    original_data = [d for d in ori_data['train']['text']]
    synthetic_data = [d for d in syn_data['train']['text']]

    label_column_index = ['label1', 'label2']

    original_labels = ["\t".join(
        [line[idx] for idx in label_column_index]) for line in ori_data['train']]
    synthetic_labels = ["\t".join(
        [line[idx] for idx in label_column_index]) for line in syn_data['train']]

    model = SentenceTransformer(args.model_name_or_path)
    model.eval()

    with torch.no_grad():
        synthetic_embeddings = []
        for i in tqdm(range(len(synthetic_data) // args.batch_size)):
            embeddings = model.encode(
                synthetic_data[i * args.batch_size:(i + 1) * args.batch_size])
            synthetic_embeddings.append(embeddings)

    synthetic_embeddings = np.concatenate(synthetic_embeddings)
    log_embeddings(synthetic_embeddings, synthetic_labels[:len(synthetic_embeddings)],  # remember to check the lables size
                   os.path.join('generations', args.model_name_or_path),
                   args.synthetic_file.split("/")[1].split(".")[0])

    with torch.no_grad():
        original_embeddings = []
        for i in tqdm(range(len(original_data) // args.batch_size)):
            embeddings = model.encode(
                original_data[i * args.batch_size:(i + 1) * args.batch_size])
            original_embeddings.append(embeddings)

    original_embeddings = np.concatenate(original_embeddings)
    log_embeddings(original_embeddings, original_labels[:len(original_embeddings)],
                   os.path.join('generations', args.model_name_or_path),
                   args.original_file.split("/")[1].split(".")[0])


if __name__ == "__main__":
    main()
