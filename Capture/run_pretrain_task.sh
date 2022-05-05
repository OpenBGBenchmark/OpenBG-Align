#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python pretrain_task.py \
 --output_dir ./testv2 \
 --from_pretrained ./pytorch_model_8.bin \
 --bert_model bert-base-chinese \
 --config_file ./config/capture.json\
 --predict_feature \
 --learning_rate 1e-4 \
 --train_batch_size 64 \
 --max_seq_length 36 \
 --lmdb_file ../testv1/item_valid_image_feature.lmdb \
 --caption_path ../../item_valid_info.jsonl \
 --save_name capture_subset_v2_MLM_MRM_CLR \
 --MLM \
 --MRM \
 --CLR