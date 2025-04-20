import paddle
from paddle.io import Dataset, DataLoader, BatchSampler, DistributedBatchSampler
import numpy as np
import os
import time
from datetime import date
from tqdm import tqdm
from utils import *
from model import *
import sys
sys.stdout.reconfigure(encoding='utf-8')
import argparse

# 配置文件
class Args():
    def __init__(self):
        self.dataset_dir = "data/data323258/w_data" # root directory containing the datasets
        self.unitid_file = "unitid.txt"
        self.train_file = "1w_train.txt"
        self.test_file = "test.txt"
        self.test_gt_file = "test_gt.txt"
        self.batch_size = 32
        self.lr = 0.0001
        self.maxlen = 200
        self.hidden_units = 1024
        self.emb_dim = 1024
        self.num_blocks = 2
        self.num_epochs = 3
        self.num_heads = 1
        self.dropout_rate = 0.2
        self.device = "gpu"
        self.inference_only = True
        self.state_dict_path = "checkpoint/SASRec.epoch=3.lr=0.0001.layer=2.head=1.hidden=1024.maxlen=200.pth"

args = Args()
parser = argparse.ArgumentParser(description="This is a description of the script.")
parser.add_argument("--dataset_dir", type=str, help="数据集路径")
parser.add_argument("--output_path", type=str, help="输出结果路径")
args_2 = parser.parse_args()
args.dataset_dir = args_2.dataset_dir

with open(os.path.join(args.dataset_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

def infer(model):
    print("开始加载测试数据...")
    dataset = TestDataset(args)
    dataloader =  paddle.io.DataLoader(dataset, batch_size=args.batch_size,collate_fn=dataset.collate_fn)
    print("数据加载完成")
    # 全库embedding
    item_embs = paddle.to_tensor([v["embedding"] for k,v in dataset.unitid_data.items()]) 
    id2item = dict(zip([i for i in range(dataset.lenth_unit_data)],list(dataset.unitid_data.keys())))
    sf = paddle.nn.Softmax()
    with paddle.no_grad():
        with open(args_2.output_path,"w") as f:
            for user_ids,padded_embeddings, pad_mask in tqdm(dataloader):
                logits = model.predict(padded_embeddings,pad_mask,item_embs) # 全库检索
                probs = sf(logits)
                topk_values, topk_indices = paddle.topk(probs, k=10, axis=-1)
                for idx in range(topk_indices.shape[0]):
                    items = []
                    for jdx in range(topk_indices.shape[1]):
                        items.append(id2item[int(topk_indices[idx,jdx])]) # 全库检索
                    temp = [dataset.id2u[int(user_ids[idx])]," ".join([str(i) for i in items])]
                    f.write("\t".join(temp))
                    f.write("\n")
        print("Done")

model = SASRec(args).to(args.device) 

# 定义一个 XavierNormal 初始化器
xavier_normal_init = paddle.nn.initializer.XavierNormal()

for name, param in model.named_parameters():
    try:
        xavier_normal_init(param, param.block)
    except:
        print(f"{name} xaiver 初始化失败")
        pass  # 忽略初始化失败的层

epoch_start_idx = 1
if args.state_dict_path is not None:
    try:
        model.set_dict(paddle.load(args.state_dict_path))
    except:  
        print('failed loading state_dicts, pls check file path: ')
        
model.eval()
infer(model)