import paddle
from paddle.io import Dataset, DataLoader, BatchSampler, DistributedBatchSampler
import numpy as np
import os
import time
from datetime import date
from tqdm import tqdm
import random
import paddle.nn as nn
import paddle.nn.functional as F
from model import SASRec, CustomContrastiveLoss
from util import TrainDataset

class Args:
    def __init__(self):
        self.dataset_dir = "./data"
        self.train_file = "sequence_data"
        self.batch_size = 32
        self.maxlen = 50
        self.emb_dim = 1024
        self.num_workers = 4
        self.shuffle = True
        self.maxlen = 200
        self.hidden_units = 1024
        self.emb_dim = 1024
        self.num_blocks = 2
        self.num_epochs = 3
        self.num_heads = 1
        self.dropout_rate = 0.2
        self.device = "gpu"
        self.inference_only = False
        self.state_dict_path = None

def evaluate(model):
    *args, = Args()
    dataset = TrainDataset(args)
    dataloader =  paddle.io.DataLoader(dataset, batch_size=args.batch_size,collate_fn=dataset.collate_fn)
    print("数据加载完成")
    # 全库embedding
    # item_embs = paddle.to_tensor([v["embedding"] for k,v in unitid_data.items()]) 
    # sigmoid = paddle.nn.Sigmoid()
    sf = paddle.nn.Softmax()
    test_result = {"recall@5":0,"recall@10":0,"ndcg@5":0,"ndcg@10":0}
    with paddle.no_grad():
        cnt_batch = 0
        for padded_embeddings, pad_mask, gt,gt_embeddings in tqdm(dataloader):
            # logits = model.predict(padded_embeddings,pad_mask,item_embs) # 全库检索
            logits = model.predict(padded_embeddings,pad_mask,gt_embeddings) # gt向量里检索
            probs = sf(logits)
            topk_values, topk_indices = paddle.topk(probs, k=10, axis=-1)
            gt = paddle.arange(0,gt.shape[0],dtype='int64')
            def ndcg(top_k,topk_indices,gt):
                # 计算 DCG@k
                ranks = paddle.arange(1, top_k + 1, dtype='float32')  # 排名: [1, 2, ..., k]
                dcg = paddle.sum(paddle.where(topk_indices == gt.unsqueeze(-1), 1.0 / paddle.log2(ranks + 1), paddle.zeros_like(topk_indices,dtype='float32')), axis=-1)
                # IDCG 为0
                # # 计算 IDCG@k（假设正确标签在第一位）
                # idcg = 1.0 / paddle.log2(2)
                # # 计算 NDCG@k
                # ndcg = dcg / idcg
                test_result[f"ndcg@{top_k}"] += paddle.mean(dcg).item()
            def recall(top_k,topk_indices,gt):
                # 计算 recall@k
                score = paddle.sum(paddle.where(topk_indices == gt.unsqueeze(-1), 1, 0),axis=-1)
                test_result[f"recall@{top_k}"] += paddle.mean(score.astype('float32')).item()
            ndcg(10,topk_indices,gt)
            ndcg(5,topk_indices[:,:5],gt)
            recall(10,topk_indices,gt)
            recall(5,topk_indices[:,:5],gt)
            cnt_batch += 1
    test_result = {k:v/cnt_batch for k,v in test_result.items()}
    return test_result