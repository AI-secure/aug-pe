
feat_ext="stsb-roberta-base-v2"
epochs=20
result_folder="result/iclr/gpt2_stsb-roberta-base-v2/14000_n0_L7_initL7_var0_openreview_rephrase_tone_rank_len448var0_t1.2"
min_token_threshold=100

python metric.py \
    --private_data_size 5000 \
    --synthetic_folder ${result_folder} \
    --run 5  \
    --min_token_threshold ${min_token_threshold} \
    --synthetic_iteration ${epochs} \
    --model_name_or_path ${feat_ext} \
    --dataset openreview \



