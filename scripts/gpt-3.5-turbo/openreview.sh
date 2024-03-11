mlm_prob=0.5
var_type="openreview_blank_fill_1_shot_word"
feat_ext="stsb-roberta-base-v2"
length=1000
temperature=1.2
num_seed_samples=2000
lookahead_degree=0
k=3 # number of variations
L=$((k+1))
init_L=${L}
num_samples=$((L*num_seed_samples))
echo generating $num_samples samples
epochs=10
select_syn_mode=rank
model_type="gpt-3.5-turbo"
noise=0 #[11.60, 6.22, 3.38, 0] 
cls_batch_size=32
feature_extractor_batch_size=1024
api="AzureGPT"
word_var_scale=60
max_token_word_scale=5

args="--dynamic_len --word_var_scale ${word_var_scale} --max_token_word_scale ${max_token_word_scale}"
result_folder="result/iclr/${model_type}_${feat_ext}/${num_samples}_n${noise}_L${L}_initL${init_L}_var${lookahead_degree}_${var_type}_${select_syn_mode}_len${length}var${word_var_scale}wo2to${max_token_word_scale}_t${temperature}"


echo ${result_folder}

### load datacheckpoint 
data_checkpoint_args=""
for  (( iter=0; iter<=epochs; iter++ ))
do
train_file=${result_folder}/${iter}/samples.csv
if [ -e "$train_file" ]; then
    echo "$train_file does exist."
    # load from  data checkpoint
    data_checkpoint_args="--data_checkpoint_step ${iter} --data_checkpoint_path ${result_folder}/${iter}/samples.csv"
else
    echo "$train_file does not exist."
fi
done
echo load data from ${data_checkpoint_args} ${args}

## run PE
python main.py ${args} ${data_checkpoint_args} \
--train_data_file "data/openreview/iclr23_reviews_train.csv" \
--dataset "openreview" \
--api ${api} \
--model_type ${model_type} \
--do_sample  \
--noise ${noise} \
--length ${length} \
--temperature ${temperature} \
--select_syn_mode ${select_syn_mode} \
--num_samples_schedule ${num_samples} \
--combine_divide_L ${L} \
--init_combine_divide_L ${init_L} \
--variation_degree_schedule ${mlm_prob} \
--feature_extractor_batch_size ${feature_extractor_batch_size} \
--lookahead_degree ${lookahead_degree} \
--epochs ${epochs} \
--feature_extractor ${feat_ext} \
--variation_type ${var_type} \
--result_folder ${result_folder} \
--log_online \
--train_data_embeddings_file "result/embeddings/${feat_ext}/openreview_train_all.embeddings.npz"  


num_train_epochs=10
max_seq_length=512
model="roberta-base"
min_token_threshold=100
item=${result_folder}

## calculate acc 
for seed in 0 1 2 
do
for label in "label2" "label1"
do
for  (( iter=epochs; iter>=0; iter-- ))
do
train_file="${item}/${iter}/samples.csv"
echo $train_file
if [ -e "$train_file" ]; then
    echo "$train_file does exist."
    output_dir=${item}/${iter}/${label}_ep${num_train_epochs}_${model}_seq${max_seq_length}_seed${seed}/
    if [ -e "${output_dir}test_${num_train_epochs}.0_results.json" ]; then
        echo ${output_dir}test_${num_train_epochs}.0_results.json" does exist. -- SKIP running classification"
    else
        echo ${output_dir}test_${num_train_epochs}.0_results.json" does not exist. -- RUN running classification"
        python utility_eval/run_classification.py \
            --report_to none --clean_dataset  --min_token_threshold ${min_token_threshold} \
            --model_name_or_path  ${model} \
            --output_dir ${output_dir} \
            --train_file ${train_file} \
            --validation_file data/openreview/iclr23_reviews_val.csv \
            --test_file data/openreview/iclr23_reviews_test.csv \
            --do_train --do_eval --do_predict --max_seq_length ${max_seq_length} --per_device_train_batch_size ${cls_batch_size} --per_device_eval_batch_size ${cls_batch_size} \
            --learning_rate 3e-5 --num_train_epochs ${num_train_epochs} \
            --overwrite_output_dir --overwrite_cache \
            --save_strategy epoch --save_total_limit 1 --load_best_model_at_end \
            --logging_strategy epoch \
            --seed ${seed} \
            --metric_for_best_model accuracy_all --greater_is_better True \
            --evaluation_strategy epoch --label_column_name ${label}
    fi
else
    echo "$train_file does not exist."
fi
done
done
done

python metric.py \
    --private_data_size 5000 \
    --synthetic_folder ${result_folder} \
    --run 5  \
    --min_token_threshold ${min_token_threshold} \
    --synthetic_iteration ${epochs} \
    --model_name_or_path ${feat_ext} \
    --dataset openreview \



