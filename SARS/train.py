import os
import argparse
import paddle
from paddle.io import DataLoader
from paddle.optimizer import Adam
from paddle.optimizer.lr import ExponentialDecay
from paddle.amp import auto_cast, GradScaler
from tqdm import tqdm

from util import SASRecDataset
from model import SASRecTower

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ad_data",     type=str,   default="data/ad_data")
    p.add_argument("--seq_data",    type=str,   default="data/sequence_data")
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--maxlen",      type=int,   default=50)
    p.add_argument("--hidden",      type=int,   default=128)
    p.add_argument("--heads",       type=int,   default=4)
    p.add_argument("--blocks",      type=int,   default=2)
    p.add_argument("--dropout",     type=float, default=0.2)
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--neg_k",       type=int,   default=200)
    p.add_argument("--device",      type=str,   default="gpu")
    return p.parse_args()

def main():
    args = parse_args()
    paddle.set_device(args.device)

    ds     = SASRecDataset(args.ad_data, args.seq_data, maxlen=args.maxlen)
    emb_tb = ds.get_embedding_table()
    V, H   = emb_tb.shape
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=ds.collate_fn, drop_last=True)

    model = SASRecTower(
        num_items=V, maxlen=args.maxlen,
        hidden=args.hidden, heads=args.heads,
        blocks=args.blocks, dropout=args.dropout)
    # load item emb
    model.item_emb.weight.set_value(paddle.to_tensor(emb_tb))
    model.item_emb.weight.stop_gradient = True

    optimizer = Adam(
        parameters=model.parameters(),
        learning_rate=ExponentialDecay(args.lr, gamma=0.96))
    scaler    = GradScaler()

    for ep in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {ep}")
        for seqs, tgts, mask in pbar:
            with auto_cast():
                uvec = model.encode(seqs, mask)            # [B,H]
                loss = model.nce_loss(uvec, tgts, neg_k=args.neg_k)
            scaled = scaler.scale(loss)
            scaled.backward()
            scaler.minimize(optimizer, scaled)
            optimizer.clear_grad()
            pbar.set_postfix(loss=float(loss))

        paddle.save(model.state_dict(), f"sasrec_ep{ep}.pdparams")
        print("Saved sasrec_ep", ep)

if __name__=="__main__":
    main()