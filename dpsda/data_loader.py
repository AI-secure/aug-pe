import numpy as np
import logging
import collections
import csv
from datasets import load_dataset


def sample_dataset(data_name, dataset, label_column_name='label1', sample_size=5000, subsample_one_class=False):
    if subsample_one_class == False and sample_size < 0:
        return dataset
    training_dataset = dataset['train']
    sample_indices = []
    if subsample_one_class:
        if data_name == "yelp":
            label1 = 'Business Category: Restaurants'
            label2 = 'Review Stars: 5.0'
            indices = np.where((np.array(training_dataset['label1']) == label1) & (
                np.array(training_dataset['label2']) == label2))[0]
        elif data_name == "openreview":
            area = "Area: Social Aspects of Machine Learning (eg, AI safety, fairness, privacy, interpretability, human-AI interaction, ethics)"
            recommendation = "Recommendation: 8: accept, good paper"
            indices = np.where((np.array(training_dataset['label1']) == area) & (
                np.array(training_dataset['label2']) == recommendation))[0]
            logging.info(f'indices {len(indices)}')
            if sample_size < 0:
                sample_indices = indices
            else:
                sample_indices = np.random.choice(
                    indices, size=sample_size, replace=False)
                np.random.shuffle(sample_indices)
        elif data_name == "pubmed":
            indices = list(range(len(training_dataset)))
        else:
            raise ValueError(f'Unknown dataset name {dataset}')
        if sample_size < 0:
            sample_indices = indices
        else:
            sample_indices = np.random.choice(
                indices, size=sample_size, replace=False)
            np.random.shuffle(sample_indices)
    else:
        if data_name == "pubmed" or data_name == "openreview":  # random sample
            indices = list(range(len(training_dataset)))
            sample_indices = np.random.choice(
                indices, size=sample_size, replace=False)
            np.random.shuffle(sample_indices)
        else:  # random sample based on label
            label_list = training_dataset.unique(label_column_name)
            for label in label_list:
                indices = np.where(
                    np.array(training_dataset[label_column_name]) == label)[0]
                sample_num = round(
                    sample_size * (len(indices)/len(training_dataset)))
                sample_indices.append(np.random.choice(
                    indices, size=sample_num, replace=False))
            sample_indices = np.concatenate(sample_indices)
            np.random.shuffle(sample_indices)
    print(sample_indices)
    training_dataset = training_dataset.select(sample_indices)
    dataset['train'] = training_dataset
    return dataset


def load_dataset_with_special(data_file, gen):
    if gen:
        try:  # in case there are some special characters in the text
            raw_datasets = load_dataset(
                "csv", data_files=data_file, quoting=csv.QUOTE_NONE,  quotechar='', escapechar='\\')
        except:
            raw_datasets = load_dataset("csv", data_files=data_file)
    else:
        raw_datasets = load_dataset("csv", data_files=data_file)
    return raw_datasets


def load_data(dataset="yelp", data_file="data/yelp/train.csv", num_samples=-1, subsample_one_class=False, gen=False):
    print("data_file", data_file)
    if dataset == "yelp":
        prompt_counter = collections.Counter()
        raw_datasets = load_dataset_with_special(data_file, gen)
        original_data = sample_dataset(dataset, raw_datasets, label_column_name='label1',
                                       sample_size=num_samples, subsample_one_class=subsample_one_class)
        prompt_idexer = dict()

        label_column_index = ['label1', 'label2']
        for i, line in enumerate(original_data['train']):
            prompt = "\t".join([line[idx] for idx in label_column_index])
            prompt_counter[prompt] += 1

            if prompt not in prompt_idexer.keys():
                prompt_idexer[prompt] = [i]
            else:
                prompt_idexer[prompt].append(i)

        train_data = [d for d in original_data['train']['text']]
        train_labels = ["\t".join([line[idx] for idx in label_column_index])
                        for line in original_data['train']]

        return train_data, train_labels, prompt_counter, prompt_idexer
    elif dataset == "openreview":

        prompt_counter = collections.Counter()

        raw_datasets = load_dataset_with_special(data_file, gen)
        original_data = sample_dataset(dataset, raw_datasets, label_column_name='label2',
                                       sample_size=num_samples, subsample_one_class=subsample_one_class)
        prompt_idexer = dict()

        train_data = []
        train_labels = []
        for i, line in enumerate(original_data['train']):
            prompt = f"{line['label1']}\t{line['label2']}"
            prompt_counter[prompt] += 1
            if prompt not in prompt_idexer.keys():
                prompt_idexer[prompt] = [i]
            else:
                prompt_idexer[prompt].append(i)
            train_data.append(line['text'])
            train_labels.append(prompt)
        return train_data, train_labels, prompt_counter, prompt_idexer
    elif dataset == "pubmed":
        prompt_counter = collections.Counter()
        raw_datasets = load_dataset_with_special(data_file, gen)
        original_data = sample_dataset(dataset, raw_datasets, label_column_name='',
                                       sample_size=num_samples, subsample_one_class=subsample_one_class)
        prompt_idexer = dict()
        train_data = []
        train_labels = []
        for i, line in enumerate(original_data['train']):
            prompt = f"pubmed"
            prompt_counter[prompt] += 1
            if prompt not in prompt_idexer.keys():
                prompt_idexer[prompt] = [i]
            else:
                prompt_idexer[prompt].append(i)
            train_data.append(line['text'])
            train_labels.append(prompt)
        return train_data, train_labels, prompt_counter, prompt_idexer

    else:
        raise ValueError(f'Unknown dataset name {dataset}')
