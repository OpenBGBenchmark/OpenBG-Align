'''
输入商品图片+文本信息，产出embedding
'''

FIELDNAMES = ['item_id', 'features']
import csv
import logging
import io, os
import random
import math
from time import strftime, gmtime
from timeit import default_timer as timer
import sys
import torch.nn.functional as F
sys.path.append("./")

import numpy as np
from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import time

from pytorch_pretrained_bert.tokenization import BertTokenizer

from capture.datasets import Pretrain_DataSet_Train
from capture.capture import CaptureForMultiModalPreTraining_SimCLR, BertConfig
import torch.distributed as dist

from config.config_dict import capture_config_json_dict, bert_weight_name, odps_config
import pdb
from utils_args import get_args

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def load_pytorch_model(model_path,model,strict=False):
    tmp_model = torch.load(model_path)
    if hasattr(tmp_model,"module"):
        model.load_state_dict(tmp_model.module, strict=strict)
    else:
        model.load_state_dict(tmp_model, strict=strict)
    return model

def main():
    args = get_args()
    print(args)
    print(args)
    # if args.save_name is not '':
    #     timeStamp = args.save_name
    # else:
    #     timeStamp = strftime("%d-%b-%y-%X-%a", gmtime())
    #     timeStamp += "_{:0>6d}".format(random.randint(0, 10e6))

    # savePath = os.path.join(args.output_dir)
    #
    # if not os.path.exists(savePath):
    #     os.makedirs(savePath)

    # config = BertConfig.from_json_file(args.config_file)
    capture_config_json_dict["v_feature_size"] = 2048
    config = BertConfig.from_dict(capture_config_json_dict)

    if args.freeze > config.t_biattention_id[0]:
        config.fixed_t_layer = config.t_biattention_id[0]

    if args.without_coattention:
        config.with_coattention = False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    print("args.train_batch_size: ", args.train_batch_size)
    # buffer_size=args.train_batch_size*args.nraws_multiple
    # print("buffer size: ",buffer_size)
    Test_Dataset = Pretrain_DataSet_Train(
        tokenizer,
        seq_len=args.max_seq_length,
        predict_feature=args.predict_feature,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        lmdb_file=args.lmdb_file,
        caption_path=args.caption_path,
        MLM=args.MLM,
        MRM=args.MRM,
        ITM=args.ITM
    )

    num_train_optimization_steps = (
            int(
                math.ceil(Test_Dataset.num_dataset
                          / args.train_batch_size)
                / args.gradient_accumulation_steps
            )
            * (args.num_train_epochs - args.start_epoch)
    )

    default_gpu = True
    if dist.is_available() and args.distributed:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    # if args.predict_feature:
    config.v_target_size = 2048
    config.predict_feature = True
    # else:
    #     config.v_target_size = 1601
    #     config.predict_feature = False
    # config.predict_feature = True

    model = CaptureForMultiModalPreTraining_SimCLR(config)
    model = load_pytorch_model(args.from_pretrained,model)
    model.cuda()
    n_gpu = torch.cuda.device_count()
    if args.fp16:
        model.half()
    if args.local_rank != -1:
        try:
            from apex import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    print("Prepare to generate feature! ready!")
    logger.info("***** Running Testing *****")
    logger.info("  Num examples = %d", Test_Dataset.num_dataset)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    startIterID = 0
    global_step = 0
    masked_loss_v_tmp = 0
    masked_loss_t_tmp = 0
    next_sentence_loss_tmp = 0
    loss_tmp = 0
    start_t = timer()


    query_vil_id = []
    tsvfile = open(os.path.join(args.output_dir, 'item_features.tsv'), 'w', encoding='utf-8')
    writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)

    step=0
    item_ids = []
    capture_features = []
    for setp, batch in enumerate(tqdm(Test_Dataset)):
        image_ids = batch[-1]
        batch = tuple(t.cuda(non_blocking=True) for t in batch[:-1])

        input_ids, input_mask, segment_ids, lm_label_ids, is_next, \
        image_feat, image_loc, image_target, image_label, image_mask = (
            batch
        )
        # print("image_feat size", image_feat.size())

        if torch.sum(torch.isnan(image_feat)) > 0:
                print("is nan")
                continue

        query_vil_id += list(image_ids)
        masked_loss_t, masked_loss_v, next_sentence_loss, \
        pooled_output_t, pooled_output_v = model(
            input_ids,
            image_feat,
            image_loc,
            segment_ids,
            input_mask,
            image_mask,
            lm_label_ids,
            image_label,
            image_target,
            is_next,
            return_features=True
        )

        pooled_output_t = F.normalize(pooled_output_t, dim=-1)
        pooled_output_v = F.normalize(pooled_output_v, dim=-1)

        pooled_output_vil = pooled_output_v + pooled_output_t
        pooled_output_vil = F.normalize(pooled_output_vil, dim=-1).detach().cpu().numpy()

        item_ids.extend(image_ids)
        capture_features.extend(pooled_output_vil)

        if len(capture_features)%20000==0:
            tmp_Vilbert_feature = []
            for id, vil_feature in zip(item_ids, capture_features):
                tmp_Vilbert_feature.append({"item_id": str(id),
                    "features": ",".join(str(each_vil_feature) for each_vil_feature in vil_feature)}
                )
            writer.writerows(tmp_Vilbert_feature)
            item_ids = []
            capture_features = []
    if len(capture_features) > 0:
        tmp_Vilbert_feature = []
        for id, vil_feature in zip(item_ids, capture_features):
            tmp_Vilbert_feature.append({"item_id": str(id),
                                        "features": ",".join(str(each_vil_feature) for each_vil_feature in vil_feature)}
                                       )
        writer.writerows(tmp_Vilbert_feature)

    tsvfile.close()


if __name__ == "__main__":
    main()