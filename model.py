import paddle
from paddle.io import Dataset, DataLoader, BatchSampler, DistributedBatchSampler
import numpy as np
import os
import time
from datetime import date
from tqdm import tqdm
import pdb
import paddle.nn as nn
import paddle.nn.functional as F

class CustomContrastiveLoss(nn.Layer):
    def __init__(self):
        super(CustomContrastiveLoss, self).__init__()

    def forward(self, logits, labels, pad_mask, ad_idxs):
        batch_size, seq_len, dim = logits.shape
        logits_flatten = paddle.reshape(logits,[batch_size * seq_len,dim])
        labels_flatten = paddle.reshape(labels,[batch_size * seq_len,dim])
        pad_mask = paddle.reshape(pad_mask,[batch_size * seq_len])
        ad_idxs = paddle.reshape(ad_idxs,[batch_size * seq_len])
        
        # 计算相似度矩阵
        similarity_matrix = paddle.matmul(logits_flatten, labels_flatten, transpose_y=True)
        # mask
        mask = paddle.zeros(shape=[batch_size * seq_len,batch_size * seq_len],dtype='float32')
        mask = paddle.where(pad_mask == 0,mask,paddle.to_tensor(1.0,dtype='float32')) # 纵行
        mask = paddle.where(paddle.expand(pad_mask.unsqueeze(-1),shape=[batch_size * seq_len,batch_size * seq_len]) == 0,paddle.to_tensor(0.0,dtype='float32'),paddle.to_tensor(1.0,dtype='float32')) # 横行
        similarity_matrix = similarity_matrix * mask
        sf = paddle.nn.Softmax()
        similarity_matrix = sf(similarity_matrix)
        # loss
        label = (ad_idxs.unsqueeze(0) == ad_idxs.unsqueeze(-1))
        label = paddle.where(label,paddle.to_tensor(1.0,dtype='float32'),paddle.to_tensor(0.0,dtype='float32'))
        label = paddle.where(mask == 0,paddle.to_tensor(0.0,dtype='float32'),label)
        loss = paddle.where(label==paddle.to_tensor(1.0,dtype='float32'),-paddle.log2(similarity_matrix),paddle.to_tensor(0.0,dtype='float32'))
        loss_sum = paddle.sum(loss,axis=-1)
        # 返回平均损失
        return loss_sum.mean()

class PointWiseFeedForward(paddle.nn.Layer):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = paddle.nn.Conv1D(in_channels=hidden_units, out_channels=hidden_units, kernel_size=1)
        self.dropout1 = paddle.nn.Dropout(dropout_rate)
        self.relu = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv1D(in_channels=hidden_units, out_channels=hidden_units, kernel_size=1)
        self.dropout2 = paddle.nn.Dropout(dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose([0, 2, 1]))))))
        outputs = outputs.transpose([0, 2, 1])  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class SASRec(paddle.nn.Layer):
    def __init__(self, args):
        super(SASRec, self).__init__()
        self.dev = args.device
        self.pos_emb = paddle.nn.Embedding(num_embeddings=args.maxlen+1, embedding_dim=args.hidden_units, padding_idx=0)
        self.emb_dropout = paddle.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = paddle.nn.LayerList()  # to be Q for self-attention
        self.attention_layers = paddle.nn.LayerList()
        self.forward_layernorms = paddle.nn.LayerList()
        self.forward_layers = paddle.nn.LayerList()

        self.last_layernorm = paddle.nn.LayerNorm(normalized_shape=args.hidden_units, epsilon=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = paddle.nn.LayerNorm(normalized_shape=args.hidden_units, epsilon=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = paddle.nn.MultiHeadAttention(embed_dim=args.hidden_units, num_heads=args.num_heads, dropout=args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = paddle.nn.LayerNorm(normalized_shape=args.hidden_units, epsilon=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_units=args.hidden_units, dropout_rate=args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, seqs , mask): 
        pos = paddle.to_tensor(np.tile(np.arange(1, seqs.shape[1] + 1), [seqs.shape[0], 1]),dtype='float32').cuda()
        pos *= mask
        seqs += self.pos_emb(paddle.to_tensor(pos, dtype='int64').cuda())
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~paddle.tril(paddle.ones((tl, tl), dtype='bool'), diagonal=0)

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self,seqs, mask, pos_seqs, neg_seqs):  # for training        
        logits = self.log2feats(seqs,mask)  
        return logits  # B * S * D

    def predict(self, seqs, mask, item_embs):  # for inference
        log_feats = self.log2feats(seqs,mask)  

        final_feat = log_feats[:, -1, :]  

        logits = paddle.matmul(final_feat,item_embs,transpose_y=True)
        return logits  # preds  # (U, I)