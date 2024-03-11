import argparse

from apis import get_api_class_from_name


def str2bool(v):
    # From:
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--api',
        type=str,
        required=True,
        choices=['HFGPT',  'AzureGPT'],
        help='Which foundation model API to use')

    parser.add_argument(
        '--dataset',
        type=str,
        default='yelp')

    parser.add_argument(
        '--train_data_file',
        type=str,
        default='data/yelp/train.csv')

    parser.add_argument(
        '--train_data_embeddings_file',
        type=str,
        default='')
    parser.add_argument(
        '--data_checkpoint_path',
        type=str,
        default='')

    parser.add_argument(
        '--data_checkpoint_step',
        type=int,
        default=-1,
        help='Iteration of the data checkpoint')

    parser.add_argument(
        '--num_private_samples',
        type=int,
        default=-1,
        help='Number of private samples to load')

    parser.add_argument(
        '--result_folder',
        type=str,
        default='result',
        help='Folder for storing results')

    parser.add_argument(
        '--feature_extractor_batch_size',
        type=int,
        default=1024,
        help='Batch size for feature extraction')

    parser.add_argument(
        '--feature_extractor',
        type=str,
        default='all-mpnet-base-v2',
        choices=["sentence-t5-xl", "sentence-t5-large",  "sentence-t5-base",
                 "all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2", "all-mpnet-base-v2", "stsb-roberta-base-v2",
                 "roberta-large-nli-stsb-mean-tokens", "distilbert-base-nli-stsb-mean-tokens", 'text-embedding-ada-002'],
        help='Which image feature extractor to use')

    parser.add_argument(
        '--noise_multiplier',
        type=float,
        default=0.0,
        help='Noise multiplier for DP NN histogram')
    parser.add_argument(
        '--lookahead_degree',
        type=int,
        default=1,
        help=('Lookahead degree for computing distances between private and '
              'generated images'))

    # new arguments for combining the two mechanisms
    parser.add_argument(
        "--donnot_keep_last_iter",
        action="store_true",
        help=('defalut(false),  (L-1) * 1/L syn var;  1/L for syn nonvar;   \
                true:  (L) *  1/L syn var'))
    parser.add_argument(
        '--combine_divide_L',
        type=int,
        default=1,
        help=(''))
    parser.add_argument(
        '--init_combine_divide_L',
        type=int,
        default=1,
        help=(''))

    parser.add_argument(
        "--lookahead_self",
        action="store_true"
    )

    parser.add_argument(
        '--num_nearest_neighbor',
        type=int,
        default=1,
        help='Number of nearest neighbors to find in DP NN histogram')
    parser.add_argument(
        '--nn_mode',
        type=str,
        default='L2',
        choices=['L2', 'IP', 'cos_sim'],
        help='Which distance metric to use in DP NN histogram')

    parser.add_argument(
        '--count_threshold',
        type=float,
        default=0.0,
        help='Threshold for DP NN histogram')

    parser.add_argument(
        '--compute_fid',
        type=str2bool,
        default=True,
        help='Whether to compute FID')

    parser.add_argument(
        '--num_samples_schedule',
        type=str,
        default='1000,'*9 + '1000',
        help='Number of samples to generate at each iteration')
    parser.add_argument(
        '--variation_degree_schedule',
        type=str,
        default='0,'*9 + '0',
        help='Variation degree at each iteration')

    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='Number of epochs')
    parser.add_argument(
        "--subsample_one_class",
        action="store_true"
    )

    parser.add_argument(
        "--select_syn_mode",
        type=str,
        default='rank',
        choices=['prob', 'rank'],
        help='sample synthetic data from the histogram by top ranking or by probability')

    parser.add_argument(
        "--save_syn_mode",
        type=str,
        default='selected',
        choices=['selected', 'all', 'one_var'],
        help='save all or selected syn samples')

    parser.add_argument(
        "--log_online",
        action="store_true"
    )
    parser.add_argument('--wandb_key',       default='',
                        type=str,   help='API key for W&B.')
    parser.add_argument('--project',         default='text-API',       type=str,
                        help='Name of the project - relates to W&B project names. In --savename default setting part of the savename.')

    args, api_args = parser.parse_known_args()

    if len(list(map(int, args.num_samples_schedule.split(',')))) == 1:
        args.num_samples_schedule = [
            list(map(int, args.num_samples_schedule.split(',')))[0]]*(args.epochs+1)
    else:
        args.num_samples_schedule = list(
            map(int, args.num_samples_schedule.split(',')))

    variation_degree_type = (
        float if '.' in args.variation_degree_schedule else int)

    if len(list(map(variation_degree_type, args.variation_degree_schedule.split(',')))) == 1:

        args.variation_degree_schedule = [list(map(
            variation_degree_type, args.variation_degree_schedule.split(',')))[0]]*(args.epochs+1)
    else:
        args.variation_degree_schedule = list(
            map(variation_degree_type, args.variation_degree_schedule.split(',')))
    print(len(args.num_samples_schedule), len(args.variation_degree_schedule))
    if len(args.num_samples_schedule) != len(args.variation_degree_schedule):
        raise ValueError('The length of num_samples_schedule and '
                         'variation_degree_schedule should be the same')

    api_class = get_api_class_from_name(args.api)
    api = api_class.from_command_line_args(api_args)
    if len(args.wandb_key) == 0:  # no wandb key provided
        args.log_online = False

    print(args)

    return args, api
