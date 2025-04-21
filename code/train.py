#!/usr/bin/env python3
import os
import argparse
import paddle
from paddle.io import DataLoader
from paddle.optimizer import Adam
from paddle.optimizer.lr import ExponentialDecay
from tqdm import tqdm

from util import SASRecDataset
from model import SASRec, CrossEntropyLossIgnorePad

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ad_data",    type=str,   default="data/ad_data")
    p.add_argument("--seq_data",   type=str,   default="data/sequence_data")
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--maxlen",     type=int,   default=50)
    # 不再从 args 中读 hidden，改为自动匹配 embedding 维度
    p.add_argument("--heads",      type=int,   default=4)
    p.add_argument("--blocks",     type=int,   default=2)
    p.add_argument("--dropout",    type=float, default=0.2)
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--device",     type=str,   default="gpu")
    p.add_argument("--out_dir",    type=str,   default="./checkpoints")
    return p.parse_args()

def main():
    args = parse_args()
    paddle.set_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Dataset & DataLoader
    ds     = SASRecDataset(args.ad_data, args.seq_data, maxlen=args.maxlen)
    emb_tb = ds.get_embedding_table()                # numpy [V, D]
    num_items, emb_dim = emb_tb.shape                # V, D
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,      # Windows/MacOS 下设为 0
        collate_fn=ds.collate_fn)

    # 2) Model & Loss & Optimizer
    model = SASRec(
        num_items=num_items,
        maxlen=args.maxlen,
        hidden=emb_dim,           # 关键：用广告 emb_dim 作为 hidden
        heads=args.heads,
        blocks=args.blocks,
        dropout=args.dropout
    )
    # 加载预训练 item embedding
    model.item_emb.weight.set_value(
        paddle.to_tensor(emb_tb, dtype='float32'))
    model.item_emb.weight.stop_gradient = True

    criterion = CrossEntropyLossIgnorePad(ignore_idx=0)
    scheduler = ExponentialDecay(learning_rate=args.lr, gamma=0.96, verbose=True)
    optimizer = Adam(parameters=model.parameters(),
                     learning_rate=scheduler)

    # 3) Training
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for seqs, tgts, mask in pbar:
            logits = model(seqs, mask)          # [B, T, V]
            loss   = criterion(logits, tgts, mask)

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            pbar.set_postfix(loss=float(loss))

        ckpt = os.path.join(args.out_dir, f"sasrec_ep{epoch}.pdparams")
        paddle.save(model.state_dict(), ckpt)
        print("Saved checkpoint:", ckpt)

if __name__=="__main__":
    main()