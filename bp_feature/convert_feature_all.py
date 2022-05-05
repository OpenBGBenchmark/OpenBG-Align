import os
import numpy as np
from tensorpack.dataflow import RNGDataFlow, PrefetchDataZMQ
import json
import csv
import re
import traceback
from tensorpack.dataflow.serialize import LMDBSerializer

FIELDNAMES = ['item_id', 'image_h', 'image_w', 'num_boxes', 'boxes', 'features', 'cls_prob', 'title']
import sys
import pandas as pd
import zlib
import base64

csv.field_size_limit(sys.maxsize)


def read_json(file):
    f = open(file, "r", encoding="utf-8").read()
    return json.loads(f)


def write_json(file, data):
    f = open(file, "w", encoding="utf-8")
    json.dump(data, f, indent=2, ensure_ascii=False)
    return


def _file_name(row):
    return "%s/%s" % (row['folder'], (zlib.crc32(row['url'].encode('utf-8')) & 0xffffffff))

def decode_base64(data, altchars=b'+/'):
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    data = data.encode('utf-8')
    data = re.sub(rb'[^a-zA-Z0-9%s]+' % altchars, b'', data)  # normalize
    missing_padding = len(data) % 4
    if missing_padding:
        data += b'='* (4 - missing_padding)
    return base64.urlsafe_b64decode(data)

class Conceptual_Caption(RNGDataFlow):
    """
    """

    def __init__(self, input_file, data_len, shuffle=False):
        """
        Same as in :class:`ILSVRC12`.
        """
        self.shuffle = shuffle
        self.num_file = 30

        self.infiles = [input_file]
        self.counts = []
        self.data_len = data_len

    def __len__(self):
        return self.data_len

    def __iter__(self):
        cnt = 0
        for infile in self.infiles:
            count = 0
            with open(infile, encoding='utf-8', mode='r') as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
                for item in reader:
                    cnt += 1

                    try:
                        # print(item)
                        item_id = item['item_id']
                        image_h = int(item['image_h'])
                        image_w = int(item['image_w'])
                        num_boxes = item['num_boxes']
                        boxes = np.array(json.loads(item['boxes']), dtype=np.float32).reshape(int(num_boxes), 4)
                        features = np.array(json.loads(item['features']), dtype=np.float32).reshape(int(num_boxes), 2048)
                        caption = item["title"]
                    except Exception as e:
                        traceback.print_exc()
                        print("error: ", e, item_id)
                        continue
                    yield [features, boxes, num_boxes, image_h, image_w, item_id, caption]
            print(cnt)


if __name__ == '__main__':
    # 商品info文件 + bp_feature文件 + lmdb保存文件
    input_file = '../item_valid_info.jsonl'
    bp_feature_input_file = './testv1/item_valid_image_feature.csv'
    bp_feature_lmdb_file = './testv1/item_valid_image_feature.lmdb'

    table_data = []
    with open(input_file, encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            item_id = line['item_id']
            table_data.append(item_id)
    data_len = len(table_data)

    # time.sleep(25200)
    ds = Conceptual_Caption(bp_feature_input_file, data_len)
    ds1 = PrefetchDataZMQ(ds)

    LMDBSerializer.save(ds1, bp_feature_lmdb_file)



