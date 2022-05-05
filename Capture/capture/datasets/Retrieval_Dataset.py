import copy
import json
import logging
import os
import random
import base64
from odps import ODPS
import csv
import numpy as np
import tensorpack.dataflow as td
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import sys
import pdb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test_important example for the language model."""

    def __init__(
        self,
            image_feat=None,
            image_target=None,
            caption=None,
            is_next=None,
            lm_labels=None,
            image_loc=None,
            num_boxes=None,
            other_image_feat=None,
            other_image_target=None,
            other_caption=None,
            other_is_next=None,
            other_lm_labels=None,
            other_image_loc=None,
            other_num_boxes=None,
            label=None
    ):
        self.image_feat = image_feat
        self.caption = caption
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model
        self.image_loc = image_loc
        self.image_target = image_target
        self.num_boxes = num_boxes
        self.other_image_feat = other_image_feat
        self.other_caption = other_caption
        self.other_is_next = other_is_next  # nextSentence
        self.other_lm_labels = other_lm_labels  # masked words for language model
        self.other_image_loc = other_image_loc
        self.other_image_target = other_image_target
        self.other_num_boxes = other_num_boxes
        self.label=label

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(
            self,
            input_ids=None,
            input_mask=None,
            segment_ids=None,
            is_next=None,
            lm_label_ids=None,
            image_feat=None,
            image_target=None,
            image_loc=None,
            image_label=None,
            image_mask=None,
            other_input_ids=None,
            other_input_mask=None,
            other_segment_ids=None,
            other_is_next=None,
            other_lm_label_ids=None,
            other_image_feat=None,
            other_image_target=None,
            other_image_loc=None,
            other_image_label=None,
            other_image_mask=None,
            label=None
        ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids
        self.image_feat = image_feat
        self.image_loc = image_loc
        self.image_label = image_label
        self.image_target = image_target
        self.image_mask = image_mask
        self.other_input_ids = other_input_ids
        self.other_input_mask = other_input_mask
        self.other_segment_ids = other_segment_ids
        self.other_is_next = other_is_next
        self.other_lm_label_ids = other_lm_label_ids
        self.other_image_feat = other_image_feat
        self.other_image_loc = other_image_loc
        self.other_image_label = other_image_label
        self.other_image_target = other_image_target
        self.other_image_mask = other_image_mask
        self.label=label

class Retrieval_Dataset(Dataset):
    def __init__(
            self,
            tokenizer,
            seq_len,
            predict_feature=True,
            batch_size=512,
            num_workers=25,
            lmdb_file=None,
            caption_path=None,
            label_list=None,
            MLM=False,
            MRM=False,
            ITM=False
    ):
        lmdb_file = lmdb_file
        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.length = len(ds)

        print("len: ", len(ds))

        preprocess_function = BertPreprocessBatch(
            caption_path,
            tokenizer,
            seq_len,
            36,
            predict_feature=predict_feature,
            MLM=MLM,
            MRM=MRM,
            ITM=ITM,
        )

        ds = td.MapData(ds, preprocess_function)
        self.ds = td.BatchData(ds, batch_size)
        self.ds.reset_state()  # TODO: it is retained in the original version

        self.predict_feature = predict_feature

        self.seq_len = seq_len
        self.region_len = 36
        self.tokenizer = tokenizer
        self.label_map = {label: i for i, label in enumerate(label_list)}

        self.MLM=MLM
        self.MRM=MRM
        self.ITM=ITM

    def __len__(self):
        return self.ds.size()

    def __iter__(self):
        for batch in self.ds.get_data():
            input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, \
            image_loc, image_target, image_label, image_mask, image_id, \
            other_input_ids, other_input_mask, other_segment_ids, other_lm_label_ids, other_is_next, other_image_feat, \
            other_image_loc, other_image_target, other_image_label, other_image_mask, other_image_id, \
            label = batch

            # add by myself
            # anchor
            g_image_feat = np.sum(image_feat, axis=0) / np.sum(image_mask, axis=0, keepdims=True)
            image_feat = np.concatenate([np.expand_dims(g_image_feat, axis=0), image_feat], axis=0)
            image_feat = np.array(image_feat, dtype=np.float32)
            g_image_loc = np.array([0, 0, 1, 1, 1], dtype=np.float32)
            image_loc = np.concatenate([np.expand_dims(g_image_loc, axis=0), image_loc], axis=0)
            image_loc = np.array(image_loc, dtype=np.float32)
            g_image_mask = np.array([1])
            image_mask = np.concatenate([g_image_mask, image_mask], axis=0)

            # other_
            other_g_image_feat = np.sum(other_image_feat, axis=0) / np.sum(other_image_mask, axis=0, keepdims=True)
            other_image_feat = np.concatenate([np.expand_dims(other_g_image_feat, axis=0), other_image_feat], axis=0)
            other_image_feat = np.array(other_image_feat, dtype=np.float32)
            other_g_image_loc = np.array([0, 0, 1, 1, 1], dtype=np.float32)
            other_image_loc = np.concatenate([np.expand_dims(other_g_image_loc, axis=0), other_image_loc], axis=0)
            other_image_loc = np.array(other_image_loc, dtype=np.float32)
            other_g_image_mask = np.array([1])
            other_image_mask = np.concatenate([other_g_image_mask, other_image_mask], axis=0)

            item_data = (input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat,
                         image_loc, image_target, image_label, image_mask,
                         other_input_ids, other_input_mask, other_segment_ids, other_lm_label_ids, other_is_next,
                         other_image_feat,
                         other_image_loc, other_image_target, other_image_label, other_image_mask,
                         label, image_id, other_image_id)
            yield tuple([torch.tensor(data) for data in item_data])

class BertPreprocessBatch(object):
    def __init__(
            self,
            caption_path,
            tokenizer,
            seq_len,
            region_len,
            label_map,
            split="Train",
            predict_feature=False,
            visualization=False,
            MLM=True,
            MRM=True,
            ITM=True,
    ):

        self.MLM = MLM
        self.MRM = MRM
        self.ITM = ITM

        self.split = split
        self.seq_len = seq_len
        self.region_len = region_len
        self.tokenizer = tokenizer
        self.predict_feature = predict_feature
        self.label_map = label_map

        self.id_info_dict = {}
        with open(caption_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = json.loads(line.strip())
                item_id = line['item_id']
                self.id_info_dict[item_id] = line

        self.captions = []
        for each in self.id_info_dict:
            self.captions.append(self.id_info_dict[each]["title"])
        self.num_caps = len(self.captions)
        self.visualization = visualization

    def __call__(self, data):


        image_feature_wp, image_location_wp, num_boxes, image_h, image_w, item_id, caption \
        ,other_image_feature_wp, other_image_location_wp, other_num_boxes, other_image_h, \
        other_image_w, other_item_id, other_caption, label = data
        label = self.label_map[label]

        ##########
        # anchor
        image_feature = np.zeros((self.region_len, 2048), dtype=np.float32)
        image_target = np.zeros((self.region_len, 1601), dtype=np.float32)
        image_location = np.zeros((self.region_len, 5), dtype=np.float32)

        num_boxes = int(num_boxes)
        image_feature[:num_boxes] = image_feature_wp
        image_location[:num_boxes, :4] = image_location_wp

        image_location[:, 4] = (image_location[:, 3] - image_location[:, 1]) * (
                    image_location[:, 2] - image_location[:, 0]) / (float(image_w) * float(image_h))

        image_location[:, 0] = image_location[:, 0] / float(image_w)
        image_location[:, 1] = image_location[:, 1] / float(image_h)
        image_location[:, 2] = image_location[:, 2] / float(image_w)
        image_location[:, 3] = image_location[:, 3] / float(image_h)

        if self.predict_feature:
            image_feature = copy.deepcopy(image_feature)
            image_target = copy.deepcopy(image_feature)
        else:
            image_feature = copy.deepcopy(image_feature)
            image_target = copy.deepcopy(image_target)

        # caption
        caption = caption
        caption, is_next = self.random_cap(caption)
        tokens_caption = self.tokenizer.tokenize(caption)


        ##########
        # other
        other_image_feature = np.zeros((self.region_len, 2048), dtype=np.float32)
        other_image_target = np.zeros((self.region_len, 1601), dtype=np.float32)
        other_image_location = np.zeros((self.region_len, 5), dtype=np.float32)

        other_num_boxes = int(other_num_boxes)
        other_image_feature[:other_num_boxes] = other_image_feature_wp
        other_image_location[:other_num_boxes, :4] = other_image_location_wp

        other_image_location[:, 4] = (other_image_location[:, 3] - other_image_location[:, 1]) * (
                other_image_location[:, 2] - other_image_location[:, 0]) / (float(other_image_w) * float(other_image_h))

        other_image_location[:, 0] = other_image_location[:, 0] / float(other_image_w)
        other_image_location[:, 1] = other_image_location[:, 1] / float(other_image_h)
        other_image_location[:, 2] = other_image_location[:, 2] / float(other_image_w)
        other_image_location[:, 3] = other_image_location[:, 3] / float(other_image_h)

        if self.predict_feature:
            other_image_feature = copy.deepcopy(other_image_feature)
            other_image_target = copy.deepcopy(other_image_feature)
        else:
            other_image_feature = copy.deepcopy(other_image_feature)
            other_image_target = copy.deepcopy(other_image_target)

        # caption
        other_caption = other_caption
        other_tokens_caption, other_is_next = self.random_cap(other_caption)
        other_tokens_caption = self.tokenizer.tokenize(other_caption)

        cur_example = InputExample(
            image_feat=image_feature,
            image_target=image_target,
            caption=tokens_caption,
            is_next=is_next,
            image_loc=image_location,
            num_boxes=num_boxes,
            other_image_feat=other_image_feature,
            other_image_target=other_image_target,
            other_caption=other_tokens_caption,
            other_is_next=other_is_next,
            other_image_loc=other_image_location,
            other_num_boxes=other_num_boxes,
            label=label
        )

        # transform sample to features
        cur_features = self.convert_example_to_features(cur_example,
                                                        self.seq_len,
                                                        self.tokenizer,
                                                        self.region_len,
                                                        self.label)

        cur_tensors = (
            cur_features.input_ids,
            cur_features.input_mask,
            cur_features.segment_ids,
            cur_features.lm_label_ids,
            cur_features.is_next,
            cur_features.image_feat,
            cur_features.image_loc,
            cur_features.image_target,
            cur_features.image_label,
            cur_features.image_mask,
            item_id,
            cur_features.other_input_ids,
            cur_features.other_input_mask,
            cur_features.other_segment_ids,
            cur_features.other_lm_label_ids,
            cur_features.other_is_next,
            cur_features.other_image_feat,
            cur_features.other_image_loc,
            cur_features.other_image_target,
            cur_features.other_image_label,
            cur_features.other_image_mask,
            other_item_id,
            cur_features.label
        )
        return cur_tensors

    def random_cap(self, caption):
        if self.visualization:
            return caption, 0

        if self.ITM:
            if random.random() > 0.5:
                label = 0
            else:
                caption = self.get_random_caption()
                label = 1
        else:
            label = 0

        return caption, label

    def get_random_caption(self):
        rand_doc_idx = random.randint(0, self.num_caps - 1)
        caption = self.captions[rand_doc_idx]

        return caption

    def _convert_example(self, caption, max_seq_length, tokenizer, image_feat, image_loc, num_boxes, max_region_length):
        self._truncate_seq_pair(caption, max_seq_length - 2)
        caption, caption_label = self.random_word(caption, tokenizer)

        image_feat, image_loc, image_label = self.random_region(image_feat, image_loc, num_boxes)

        # concatenate lm labels and account for CLS, SEP, SEP
        # lm_label_ids = ([-1] + caption_label + [-1] + image_label + [-1])
        lm_label_ids = [-1] + caption_label + [-1]
        # image_label = ([-1] + image_label)
        tokens = []
        segment_ids = []

        tokens.append("[CLS]")
        segment_ids.append(0)

        for token in caption:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * (len(input_ids))
        image_mask = [1] * (num_boxes)
        # Zero-pad up to the visual sequence length.
        while len(image_mask) < max_region_length:
            image_mask.append(0)
            image_label.append(-1)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(lm_label_ids) == max_seq_length
        assert len(image_mask) == max_region_length
        assert len(image_label) == max_region_length

        return input_ids, input_mask, segment_ids, lm_label_ids, image_label, image_mask

    def convert_example_to_features(self, example,
                                    max_seq_length,
                                    tokenizer,
                                    max_region_length,
                                    label):
        image_feat = example.image_feat
        caption = example.caption
        image_loc = example.image_loc
        image_target = example.image_target
        num_boxes = int(example.num_boxes)

        other_image_feat = example.other_image_feat
        other_caption = example.other_caption
        other_image_loc = example.other_image_loc
        other_image_target = example.other_image_target
        other_num_boxes = int(example.other_num_boxes)

        input_ids, input_mask, segment_ids, \
        lm_label_ids, image_label, image_mask = self._convert_example(caption,
                                                                      max_seq_length,
                                                                      tokenizer,
                                                                      image_feat,
                                                                      image_loc,
                                                                      num_boxes,
                                                                      max_region_length)

        other_input_ids, other_input_mask, other_segment_ids, \
        other_lm_label_ids, other_image_label, other_image_mask = \
            self._convert_example(other_caption,
                                  max_seq_length,
                                  tokenizer,
                                  other_image_feat,
                                  other_image_loc,
                                  other_num_boxes,
                                  max_region_length)

        features = InputFeatures(
            input_ids=np.array(input_ids),
            input_mask=np.array(input_mask),
            segment_ids=np.array(segment_ids),
            lm_label_ids=np.array(lm_label_ids),
            is_next=np.array(example.is_next),
            image_feat=image_feat,
            image_target=image_target,
            image_loc=image_loc,
            image_label=np.array(image_label),
            image_mask=np.array(image_mask),
            other_input_ids=np.array(other_input_ids),
            other_input_mask=np.array(other_input_mask),
            other_segment_ids=np.array(other_segment_ids),
            other_lm_label_ids=np.array(other_lm_label_ids),
            other_is_next=np.array(example.other_is_next),
            other_image_feat=other_image_feat,
            other_image_target=other_image_target,
            other_image_loc=other_image_loc,
            other_image_label=np.array(other_image_label),
            other_image_mask=np.array(other_image_mask),
            label=label
        )
        return features


    def _truncate_seq_pair(self, tokens_b, max_length):
        while True:
            total_length = len(tokens_b)
            if total_length <= max_length:
                break

            tokens_b.pop()

    def random_word(self, tokens, tokenizer):
        output_label = []

        if self.MLM:
            for i, token in enumerate(tokens):
                prob = random.random()
                # mask token with 15% probability

                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = "[MASK]"

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]
                    # -> rest 10% randomly keep current token
                    # append current token to output (we will predict these later)
                    try:
                        output_label.append(tokenizer.vocab[token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        output_label.append(tokenizer.vocab["[UNK]"])
                        logger.warning(
                            "Cannot find token '{}' in vocab. Using [UNK] insetad".format(token)
                        )
                else:
                    # no masking token (will be ignored by loss function later)
                    output_label.append(-1)
        else:
            for i, token in enumerate(tokens):
                output_label.append(-1)

        return tokens, output_label

    def random_region(self, image_feat, image_loc, num_boxes):
        """
        """
        output_label = []

        if self.MRM:
            for i in range(num_boxes):
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.9:
                        image_feat[i] = 0
                    output_label.append(1)
                else:
                    # no masking token (will be ignored by loss function later)
                    output_label.append(-1)
        else:
            for i in range(num_boxes):
                output_label.append(-1)

        return image_feat, image_loc, output_label



