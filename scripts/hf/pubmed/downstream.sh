
result_folder="result/pubmed/gpt2_sentence-t5-base/14000_n0_L7_initL7_var0_pubmed_rephrase_tone_rank_len448var0_t1.0"

export WANDB_DISABLED="true"


max_seq_length=512
batch_size=32
min_token_threshold=50
lr=3e-4
wd=0.01
item=${result_folder}

for model in 'bert-small' 
do
num_train_epochs=10
for  (( iter=epochs; iter>=0; iter-- ))
do
    train_file="${item}/${iter}"
    if [ -d "$train_file" ]; then
        echo "$train_file does exist."

        output_dir=${train_file}/ep${num_train_epochs}_${model}_wd${wd}lr${lr}bs${batch_size}/

        if [ -e "$output_dir/eval_results.json" ]; then
            echo "$output_dir/eval_results.json does exist. -- SKIP running classification"
        else

        echo $train_file
        python utility_eval/run_clm.py \
            --model_name_or_path prajjwal1/${model} \
            --clean_dataset  --min_token_threshold ${min_token_threshold} \
            --output_dir ${output_dir} \
            --train_file ${train_file}/samples.csv \
            --validation_file data/pubmed/dev.csv \
            --per_device_train_batch_size ${batch_size} \
            --per_device_eval_batch_size ${batch_size} \
            --learning_rate ${lr} \
            --do_eval \
            --do_train \
            --weight_decay ${wd} \
            --num_train_epochs ${num_train_epochs} \
            --save_total_limit 2 \
            --overwrite_output_dir --overwrite_cache 

        python utility_eval/run_clm.py \
            --model_name_or_path prajjwal1/${model} \
            --output_dir ${output_dir} \
            --train_file ${train_file}/samples.csv \
            --validation_file data/pubmed/test.csv \
            --per_device_train_batch_size ${batch_size} \
            --per_device_eval_batch_size ${batch_size} \
            --learning_rate ${lr} \
            --do_eval \
            --do_train \
            --weight_decay ${wd} \
            --num_train_epochs 0 
        fi
    fi
done
done

for model in 'bert-mini' 
do
num_train_epochs=20
for  (( iter=epochs; iter>=0; iter-- ))
do
    train_file="${item}/${iter}"
    if [ -d "$train_file" ]; then
        echo "$train_file does exist."

        output_dir=${train_file}/ep${num_train_epochs}_${model}_wd${wd}lr${lr}bs${batch_size}/

        if [ -e "$output_dir/eval_results.json" ]; then
            echo "$output_dir/eval_results.json does exist. -- SKIP running classification"
        else

        echo $train_file
        python utility_eval/run_clm.py \
            --model_name_or_path prajjwal1/${model} \
            --clean_dataset  --min_token_threshold ${min_token_threshold} \
            --output_dir ${output_dir} \
            --train_file ${train_file}/samples.csv \
            --validation_file data/pubmed/dev.csv \
            --per_device_train_batch_size ${batch_size} \
            --per_device_eval_batch_size ${batch_size} \
            --learning_rate ${lr} \
            --do_eval \
            --do_train \
            --weight_decay ${wd} \
            --num_train_epochs ${num_train_epochs} \
            --save_total_limit 2 \
            --overwrite_output_dir --overwrite_cache 

        python utility_eval/run_clm.py \
            --model_name_or_path prajjwal1/${model} \
            --output_dir ${output_dir} \
            --train_file ${train_file}/samples.csv \
            --validation_file data/pubmed/test.csv \
            --per_device_train_batch_size ${batch_size} \
            --per_device_eval_batch_size ${batch_size} \
            --learning_rate ${lr} \
            --do_eval \
            --do_train \
            --weight_decay ${wd} \
            --num_train_epochs 0 
        fi
    fi
done
done





