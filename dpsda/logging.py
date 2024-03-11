import logging
import os
import numpy as np
import csv
import json
from dpsda.metrics import calculate_fid, knn_precision_recall_features


def setup_logging(log_file):
    log_formatter = logging.Formatter(
        fmt=('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  '
             '%(message)s'),
        datefmt='%m/%d/%Y %H:%M:%S %p')
    root_logger = logging.getLogger()
    # root_logger.setLevel(logging.DEBUG)
    root_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)


def log_embeddings(embeddings, additional_info, folder, fname=''):
    if not os.path.exists(folder):
        os.makedirs(folder)
    savefname = os.path.join(folder, fname+'.embeddings.npz')
    print("save embeddings into", savefname)
    np.savez(
        savefname,
        embeddings=embeddings,
        additional_info=additional_info)


def load_embeddings(path):
    data = np.load(path)
    embeddings = data['embeddings']
    additional_info = data['additional_info']

    return embeddings, additional_info


def log_num_words(fname="num_word_lookahead.csv", all_gen_words=[], all_target_words=[]):
    if len(all_gen_words) == 0 or len(all_target_words) == 0:
        return
    with open(fname, 'w', newline='', encoding="utf-8") as wf:
        csv_writer = csv.writer(wf)
        csv_writer.writerow(["target", "gen", "diff"])
        diff_list = []
        diff_abs_list = []
        for i in range(len(all_target_words)):
            try:
                diff_list.append(all_gen_words[i] - all_target_words[i])
                diff_abs_list.append(
                    abs(all_gen_words[i] - all_target_words[i]))
                csv_writer.writerow(
                    [all_target_words[i], all_gen_words[i], all_gen_words[i] - all_target_words[i]])
            except:
                continue
        csv_writer.writerow(["mean_abs", "var_abs", "mean", "var"])
        csv_writer.writerow([np.mean(diff_abs_list), np.std(
            diff_abs_list), np.mean(diff_list), np.std(diff_list)])


def log_prompt_generation(fname="prompt_generation.jsonl", prompts=[], generations=[]):
    new_variants_samples = []
    for x in generations:
        new_variants_samples.extend(x.tolist())

    if len(prompts) == 0 or len(new_variants_samples) == 0:
        return
    with open(fname, "w") as file:
        for i in range(len(prompts)):
            try:
                json_str = json.dumps(
                    {"prompt": prompts[i], "generation": new_variants_samples[i]})
                file.write(json_str + "\n")
            except:
                continue


def log_count(count, clean_count, path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    title = ['type', 'count']
    with open(path, 'w', newline='', encoding="utf-8") as wf:
        csv_writer = csv.writer(wf)
        csv_writer.writerow(title)
        csv_writer.writerow(["count", count.tolist()])
        csv_writer.writerow(["clean_count", clean_count.tolist()])


def compute_fid(synthetic_features, all_private_features, feature_extractor, folder='', step=0, log_online=False):

    logging.info(
        f'Computing FID and F1 for syn shape {synthetic_features.shape}')
    fid = calculate_fid(synthetic_features, all_private_features)
    state = knn_precision_recall_features(
        ref_features=all_private_features, eval_features=synthetic_features)
    logging.info(f'fid={fid} F1={state}')
    log_fid(folder, fid, state["f1"],
            state["precision"], state["recall"], step)
    if log_online:
        import wandb
        wandb.log({f'metric/fid_{feature_extractor[:10]}': fid, }, step=step)


def log_fid(folder, fid, f1, precision, recall, t, save_fname='fid.csv'):
    with open(os.path.join(folder, save_fname), 'a') as f:
        f.write(f'{t} {fid} {f1} {precision} {recall}\n')


def log_fid_list(folder, fids, t, save_fname='fid.csv'):
    write_list = [t]
    write_list.extend(fids)
    with open(os.path.join(folder, save_fname), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(write_list)


def log_samples(samples, additional_info, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    all_data = []
    for i in range(len(samples)):
        seq = samples[i]
        labels = additional_info[i]
        if seq:
            seq = " ".join(seq.split())
            if "pubmed" in labels:
                all_data.append([seq])
            else:
                labels = labels.strip().split("\t")
                all_data.append([seq]+labels)

    if "pubmed" in additional_info[0]:  # unconditional
        title = ['text']
    else:
        title = ['text', 'label1', 'label2']
    try:
        with open(os.path.join(folder, 'samples.csv'), 'w', newline='', encoding="utf-8") as wf:
            csv_writer = csv.writer(wf)
            csv_writer.writerow(title)
            for obj in all_data:
                if obj[0]:  # remove empty sequences
                    csv_writer.writerow(obj)
    except:  # in case there are some special characters in the text
        with open(os.path.join(folder, 'samples.csv'), 'w', newline='', encoding="utf-8") as wf:
            csv_writer = csv.writer(
                wf, quoting=csv.QUOTE_NONE,  quotechar='', escapechar='\\')
            csv_writer.writerow(title)
            for obj in all_data:
                if obj[0]:  # remove empty sequences
                    csv_writer.writerow(obj)
    return all_data
