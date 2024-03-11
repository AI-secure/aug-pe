
feat_ext="sentence-t5-base"
epochs=10
result_folder="result/pubmed/gpt2_sentence-t5-base/14000_n0_L7_initL7_var0_pubmed_rephrase_tone_rank_len448var0_t1.0"
min_token_threshold=50


python metric.py \
    --private_data_size 5000 \
    --synthetic_folder ${result_folder} \
    --run 5  \
    --min_token_threshold ${min_token_threshold} \
    --synthetic_iteration ${epochs} \
    --original_file "data/pubmed/train.csv"  \
    --train_data_embeddings_file result/embeddings/${feat_ext}/pubmed_train_all.embeddings.npz \
    --model_name_or_path ${feat_ext} \
    --dataset pubmed \




