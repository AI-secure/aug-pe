mlm_prob=0.5
var_type="yelp_rephrase_tone"
feat_ext="stsb-roberta-base-v2"
length=64
temperature=1.4
num_seed_samples=5000
lookahead_degree=0
k=6 # number of variations
L=$((k+1))
init_L=${L}
num_samples=$((L*num_seed_samples))
echo generating $num_samples samples
epochs=20
word_var_scale=0
select_syn_mode=rank
model_type=gpt2  
noise=0
args=""
cls_batch_size=32
api="HFGPT"
feature_extractor_batch_size=1024
if [ "$model_type" = "gpt2-large" ]; then
    batch_size=64
elif [ "$model_type" = "gpt2-medium" ]; then
    batch_size=128
elif [ "$model_type" = "gpt2" ]; then
    batch_size=1024
else
    batch_size=8
fi
result_folder="result/yelp/${model_type}_${feat_ext}/${num_samples}_n${noise}_L${L}_initL${init_L}_var${lookahead_degree}_${var_type}_${select_syn_mode}_len${length}var${word_var_scale}_t${temperature}"

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

### run PE
python main.py ${args} ${data_checkpoint_args} \
--train_data_file "data/yelp/train.csv" \
--api ${api} \
--noise ${noise} \
--model_type ${model_type} \
--do_sample  \
--length ${length} \
--random_sampling_batch_size ${batch_size} \
--variation_batch_size ${batch_size} \
--fp16 \
--temperature ${temperature} \
--select_syn_mode ${select_syn_mode} \
--num_samples_schedule ${num_samples} \
--combine_divide_L ${L} \
--init_combine_divide_L ${init_L} \
--variation_degree_schedule ${mlm_prob} \
--lookahead_degree ${lookahead_degree} \
--epochs ${epochs} \
--use_subcategory \
--feature_extractor ${feat_ext} \
--feature_extractor_batch_size ${feature_extractor_batch_size} \
--mlm_probability ${mlm_prob} \
--variation_type ${var_type} \
--result_folder ${result_folder} \
--log_online \
--train_data_embeddings_file "result/embeddings/${feat_ext}/yelp_train_all.embeddings.npz" 


### calculate downstream model acc 
num_train_epochs=5
for seed in 0 1 2 
do
for label in "label2" "label1"
do
for  (( iter=epochs; iter>=0; iter-- ))
do
train_file=${result_folder}/${iter}/samples.csv
if [ -e "$train_file" ]; then
    echo "$train_file does exist."
    output_dir=${result_folder}/${iter}/${label}_clean_ep${num_train_epochs}_seed${seed}/
    if [ -e "${output_dir}test_${num_train_epochs}.0_results.json" ]; then
        echo "${output_dir}test_${num_train_epochs}.0_results.json  does exist. -- SKIP running classification"
    else
        echo "${output_dir}test_${num_train_epochs}.0_results.json  does not exist. -- RUN running classification"
        python utility_eval/run_classification.py \
            --report_to none --clean_dataset --model_name_or_path  roberta-base \
            --output_dir ${output_dir} \
            --train_file ${train_file} --validation_file data/yelp/dev.csv --test_file data/yelp/test.csv \
            --do_train --do_eval --do_predict --max_seq_length 512 --per_device_train_batch_size ${cls_batch_size} --per_device_eval_batch_size ${cls_batch_size} \
            --learning_rate 3e-5 --num_train_epochs ${num_train_epochs} \
            --overwrite_output_dir --overwrite_cache \
            --save_strategy epoch --save_total_limit 2 --load_best_model_at_end \
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


### calculate metric
python metric.py \
    --dataset yelp \
    --private_data_size 5000 \
    --synthetic_folder ${result_folder} \
    --run 5  \
    --synthetic_iteration ${epochs} \
    --model_name_or_path ${feat_ext} \
    
