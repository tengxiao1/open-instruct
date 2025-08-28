
# export model=sft_olmo2_lc_wildchat_8192_olmo2_lc_orz_mix_24175__1__1754343934
# export step=100
# echo "Downloading model: $model"
#     source_dir=gs://ai2-llm/post-training//saumyam//output/${model}_checkpoints/step_${step}
#     target_dir=/weka/oe-adapt-default/saumyam/model_checkpoints/${model}_checkpoints
#         if [ -d "$target_dir" ] && [ "$(ls -A "$target_dir")" ]; then
#     echo "Target directory $target_dir already exists and is not empty. Skipping $model."
#     continue
#     fi
#     mkdir -p $target_dir
#     gsutil -o GSUtil:parallel_thread_count=1 -o GSUtil:sliced_object_download_threshold=150 -m cp -r $source_dir $target_dir