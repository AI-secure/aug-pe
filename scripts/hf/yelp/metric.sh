feat_ext="stsb-roberta-base-v2"
epochs=20
result_folder="result/yelp/gpt2_stsb-roberta-base-v2/35000_n0_L7_initL7_var0_yelp_rephrase_tone_rank_len64var0_t1.4"

### calculate metric
python metric.py \
    --dataset yelp \
    --private_data_size 5000 \
    --synthetic_folder ${result_folder} \
    --run 5  \
    --synthetic_iteration ${epochs} \
    --model_name_or_path ${feat_ext} \
