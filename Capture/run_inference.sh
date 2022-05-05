#!/usr/bin/env bash
python inference.py \
 --output_dir ./testv2 \
 --from_pretrained ./pytorch_model_8.bin \
 --lmdb_file ../testv1/item_valid_image_feature.lmdb \
 --caption_path ../../item_valid_info.jsonl \
 --config_file ./config/capture.json \
 --bert_model bert-base-chinese \
 --predict_feature \
 --train_batch_size 64 \
 --max_seq_length 36