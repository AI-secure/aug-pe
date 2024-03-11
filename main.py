# from parameters import parse_args
import logging
import os
import numpy as np
from dpsda.logging import setup_logging, log_num_words, load_embeddings, log_samples, log_count, compute_fid, log_prompt_generation
from dpsda.data_loader import load_data
from dpsda.feature_extractor import extract_features
from dpsda.dp_counter import dp_nn_histogram
from dpsda.arg_utils import parse_args

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    args, api = parse_args()

    if args.log_online:
        import wandb
        _ = os.system('wandb login {}'.format(args.wandb_key))
        os.environ['WANDB_API_KEY'] = args.wandb_key
        wandb.init(project=args.project, name=args.result_folder[7:])
        wandb.config.update(args)

    if args.data_checkpoint_step >= len(args.num_samples_schedule) - 1:
        logging.info(f'finished {args.data_checkpoint_step} PE iterations!')
        exit(0)

    os.makedirs(args.result_folder, exist_ok=True)
    setup_logging(os.path.join(args.result_folder, 'log.log'))
    logging.info(f'config: {args}')
    logging.info(f'API config: {api.args}')
    # load private data
    all_private_samples, all_private_labels, private_labels_counter, private_labels_indexer = load_data(
        dataset=args.dataset,
        data_file=args.train_data_file,
        num_samples=args.num_private_samples,
        subsample_one_class=args.subsample_one_class)

    # if we randomly subsample the private data, we save the subsampled data.
    if args.num_private_samples > 0:
        log_samples(samples=all_private_samples,  additional_info=all_private_labels,
                    folder=f'{args.result_folder}/train')

    private_classes = list(private_labels_counter.keys())
    logging.info(
        f'Private_num_classes: {len(private_classes)}, Private_num_samples: {len(all_private_samples)}, Private_num_labels:{len(all_private_labels)}')

    logging.info('Extracting features of private data')
    if args.train_data_embeddings_file != '':
        logging.info(f'load features {args.train_data_embeddings_file}')
        all_private_features, all_private_labels = load_embeddings(
            args.train_data_embeddings_file)  # need to be full samples index
        all_private_samples = all_private_samples[:len(all_private_features)]
    else:
        # extract the embeddings of the private data
        all_private_features = extract_features(
            data=all_private_samples,
            batch_size=args.feature_extractor_batch_size,
            model_name=args.feature_extractor,
        )

    # Generating initial synthetic samples.
    if args.data_checkpoint_path != '':
        logging.info(
            f'Loading data checkpoint from {args.data_checkpoint_path}')
        seed_syn_samples, seed_additional_info, sync_labels_counter, sync_labels_indexer = load_data(
            dataset=args.dataset,
            data_file=args.data_checkpoint_path,
            num_samples=-1,
            gen=True,
        )  # load all samples

        if args.data_checkpoint_step < 0:
            raise ValueError('data_checkpoint_step should be >= 0')
        start_t = args.data_checkpoint_step + 1
    else:
        logging.info('Generating initial samples')
        private_lens_dict = None
        num_seed_samples = int(
            args.num_samples_schedule[0]/args.init_combine_divide_L)
        seed_syn_samples, seed_additional_info, sync_labels_counter, all_prefix_prompts = api.text_random_sampling(num_samples=num_seed_samples,
                                                                                                                   prompt_counter=private_labels_counter, lens_dict=private_lens_dict)
        os.makedirs(f'{args.result_folder}/0', exist_ok=True)
        log_prompt_generation(fname=f'{args.result_folder}/0/prompt_generation.jsonl',
                              prompts=all_prefix_prompts, generations=np.stack([seed_syn_samples], axis=1))

        if args.data_checkpoint_step >= 0:
            logging.info('Ignoring data_checkpoint_step')
        start_t = 1

    # save initial synthetic samples.
    log_samples(samples=seed_syn_samples, additional_info=seed_additional_info,
                folder=f'{args.result_folder}/{start_t-1}')
    if args.compute_fid:
        synthetic_features = extract_features(
            data=seed_syn_samples,
            batch_size=args.feature_extractor_batch_size,
            model_name=args.feature_extractor,

        )
        compute_fid(synthetic_features, all_private_features, args.feature_extractor,
                    folder=args.result_folder,  step=start_t-1, log_online=args.log_online)

    if args.init_combine_divide_L > 1:
        parent_directory = os.path.dirname(args.data_checkpoint_path)
        all_data_ckpt_path = os.path.join(
            parent_directory + "_all", 'samples.csv')
        if os.path.isfile(all_data_ckpt_path):
            logging.info(f'start to load  {all_data_ckpt_path}')
            syn_samples, additional_info, sync_labels_counter, _ = load_data(
                dataset=args.dataset,
                data_file=all_data_ckpt_path,
                num_samples=-1,
                gen=True,
            )  # load all samples
        else:
            syn_samples, additional_info = [], []
            current_idx = 0
            for class_i, class_ in enumerate(private_classes):
                num_samples_per_class = sync_labels_counter[class_]
                if num_samples_per_class == 0:
                    continue
                seed_syn_samples_per_class = seed_syn_samples[current_idx: current_idx +
                                                              num_samples_per_class]
                seed_additional_info_per_class = seed_additional_info[
                    current_idx: current_idx + num_samples_per_class]
                new_variants_samples_stacked, _, _, _, _ = api.text_variation(
                    sequences=seed_syn_samples_per_class,  # seed samples
                    additional_info=seed_additional_info_per_class,
                    num_variations_per_sequence=args.init_combine_divide_L-1,  # just do one variation
                    variation_degree=args.variation_degree_schedule[0]
                )
                syn_samples.extend(seed_syn_samples_per_class)  # seed samples
                for x in new_variants_samples_stacked:  # L-1 variations
                    syn_samples.extend(x.tolist())
                additional_info.extend(
                    seed_additional_info_per_class * args.init_combine_divide_L)
                current_idx += num_samples_per_class
                sync_labels_counter[class_] = num_samples_per_class * \
                    args.init_combine_divide_L
            log_samples(samples=syn_samples, additional_info=additional_info,
                        folder=f'{args.result_folder}/-1')
    else:
        syn_samples, additional_info = seed_syn_samples, seed_additional_info

    logging.info(
        f'initial samples size {len(syn_samples)} label {len(additional_info)}')
    for key, value in sync_labels_counter.items():
        if value > 0:
            logging.info(f'initial samples label counter {key}: {value}')

    for t in range(start_t, len(args.num_samples_schedule)):
        logging.info(f't={t}')

        if args.lookahead_degree == 0:
            packed_samples = np.expand_dims(syn_samples, axis=1)
        else:
            logging.info('Running text variation')
            packed_samples, variation_lables, all_target_words, all_gen_words, all_masked_prompts = api.text_variation(  # shape [# num_sample, # variations]
                sequences=syn_samples,
                additional_info=additional_info,
                num_variations_per_sequence=args.lookahead_degree,
                variation_degree=args.variation_degree_schedule[t])
            if args.lookahead_self:
                packed_samples = np.concatenate((packed_samples,  np.expand_dims(
                    syn_samples, axis=1)), axis=1)  # add the original samples to the variations

            os.makedirs(f'{args.result_folder}/{t}', exist_ok=True)
            log_num_words(fname=f'{args.result_folder}/{t}/num_word_lookahead.csv',
                          all_gen_words=all_gen_words, all_target_words=all_target_words)
            log_prompt_generation(fname=f'{args.result_folder}/{t}/prompt_generation.jsonl',
                                  prompts=all_masked_prompts, generations=packed_samples)

        packed_features = []
        logging.info('Running feature extraction')

        # iterate over # lookahead_degree variations.
        for i in range(packed_samples.shape[1]):
            sub_packed_features = extract_features(
                data=packed_samples[:, i],
                batch_size=args.feature_extractor_batch_size,
                model_name=args.feature_extractor,

            )
            packed_features.append(sub_packed_features)

        # take the averaged embedding for each sequence..
        packed_features = np.mean(packed_features, axis=0)
        logging.info(f'feature extraction shape {packed_features.shape}')
        logging.info('Computing histogram')
        count = []
        current_idx = 0
        # for next iteration
        new_syn_samples = []
        new_additional_info = []

        # for current iteration saving
        all_selected_samples = []
        all_selected_additional_info = []

        for class_i, class_ in enumerate(private_classes):
            # key must have the same order as  private_classes (from private_labels_counter)
            num_samples_per_class = sync_labels_counter[class_]
            if num_samples_per_class == 0:
                continue
            # get the count for each synthetic data
            public_features = packed_features[current_idx:
                                              num_samples_per_class+current_idx]
            logging.info(
                f'{class_}, {num_samples_per_class} , features shape {public_features.shape}')
            assert num_samples_per_class == public_features.shape[0]

            selected_size = int(num_samples_per_class/args.combine_divide_L)
            logging.info(f'selected_size  {selected_size}')
            if selected_size == 0:
                sub_count = []
                sub_new_indices = list(
                    range(current_idx, num_samples_per_class+current_idx))
                selected_syn_samples = [syn_samples[i]
                                        for i in sub_new_indices]
                selected_additional_info = [
                    additional_info[i] for i in sub_new_indices]
                new_variants_samples = selected_syn_samples*args.combine_divide_L
                new_variants_additional_info = selected_additional_info * args.combine_divide_L
            else:
                sub_count, sub_clean_count = dp_nn_histogram(
                    public_features=public_features,
                    private_features=all_private_features[private_labels_indexer[class_]],
                    noise_multiplier=args.noise_multiplier,
                    num_nearest_neighbor=args.num_nearest_neighbor,
                    mode=args.nn_mode,
                    threshold=args.count_threshold)
                assert np.sum(sub_count) > 0
                # Generating new indices of synthetic data
                if args.select_syn_mode == 'prob':
                    candidate_indices = np.arange(
                        current_idx, num_samples_per_class + current_idx, dtype=int)
                    sampling_prob = (sub_count) / np.sum(sub_count)
                    top_1_ind = np.argpartition(sampling_prob, -1)[-1:]
                    sub_new_indices = np.random.choice(
                        candidate_indices,
                        size=selected_size,
                        p=sampling_prob)
                    # logging.info(f'sub_new_indices size  {len(sub_new_indices)}')

                elif args.select_syn_mode == 'rank':
                    sort_index = [
                        i+current_idx for i, x in sorted(enumerate(sub_count), key=lambda x: -x[1])]
                    sub_new_indices = sort_index[:selected_size]  # top votes
                else:
                    raise ValueError(
                        f'supported select_syn_mode {args.select_syn_mode}')

                count_fname = class_.replace("\t", "_").replace(
                    " ", "_").replace("&", "").replace(":", "")
                log_count(sub_count, sub_clean_count,
                          f'{args.result_folder}/{t}/count_class/{count_fname}.csv')

                # Generate new synthetic data
                selected_syn_samples = [syn_samples[i]
                                        for i in sub_new_indices]
                selected_additional_info = [
                    additional_info[i] for i in sub_new_indices]
                # logging.info(f'selected_syn_samples shape {len(selected_syn_samples)} label {len(selected_additional_info)}')
                assert len(selected_syn_samples) == len(
                    selected_additional_info)

                new_variants_samples = []
                if args.combine_divide_L == 1:
                    _num_variations_per_sequence = 1  # just do one variation
                elif args.combine_divide_L > 1:
                    if args.donnot_keep_last_iter:
                        _num_variations_per_sequence = args.combine_divide_L
                    else:
                        _num_variations_per_sequence = args.combine_divide_L - 1
                        new_variants_samples.extend(selected_syn_samples)
                else:
                    raise ValueError('combine_divide_L should be >= 1')

                logging.info(
                    f'_num_variations_per_sequence  {_num_variations_per_sequence}')

                new_variants_samples_stacked, _, _, _, _ = api.text_variation(
                    sequences=selected_syn_samples,  # seed samples
                    additional_info=selected_additional_info,
                    num_variations_per_sequence=_num_variations_per_sequence,  # just do one variation
                    variation_degree=args.variation_degree_schedule[t]
                )

                for x in new_variants_samples_stacked:
                    new_variants_samples.extend(x.tolist())
                new_variants_additional_info = selected_additional_info * args.combine_divide_L
                # logging.info(f'new_variants_samples shape {len(new_variants_samples)} label {len(new_variants_additional_info)}')

                new_syn_samples.extend(new_variants_samples)
                new_additional_info.extend(new_variants_additional_info)
                sync_labels_counter[class_] = len(
                    new_variants_samples)  # update class size

            if args.save_syn_mode == 'selected':
                all_selected_samples.extend(selected_syn_samples)
                all_selected_additional_info.extend(selected_additional_info)
            elif args.save_syn_mode == 'one_var':
                all_selected_samples.extend(new_variants_samples_stacked[:, 0])
                all_selected_additional_info.extend(selected_additional_info)
            elif args.save_syn_mode == 'all':
                all_selected_samples.extend(
                    new_variants_samples)  # all ---  L times size
                all_selected_additional_info.extend(
                    new_variants_additional_info)

            current_idx += public_features.shape[0]

        syn_samples = new_syn_samples
        additional_info = new_additional_info
        all_data = log_samples(samples=all_selected_samples,
                               additional_info=all_selected_additional_info, folder=f'{args.result_folder}/{t}')

        if args.compute_fid:
            synthetic_features = extract_features(
                data=all_selected_samples,
                batch_size=args.feature_extractor_batch_size,
                model_name=args.feature_extractor,

            )
            compute_fid(synthetic_features, all_private_features, args.feature_extractor,
                        folder=args.result_folder,  step=t, log_online=args.log_online)
        all_data = log_samples(
            samples=syn_samples,  additional_info=additional_info, folder=f'{args.result_folder}/{t}_all')

    if args.log_online:
        wandb.finish()


if __name__ == '__main__':
    main()
