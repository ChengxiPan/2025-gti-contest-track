import os
import numpy as np
import paddle
from paddle.io import Dataset
from tqdm import tqdm

class SASRecDataset(Dataset):
    def __init__(self, ad_data_path, seq_data_path, maxlen=50):
        # 1) 加载或缓存广告 embedding 到 .npy
        map_p, embs_p = "ad_ids.npy", "ad_embs.npy"
        if not os.path.exists(map_p) or not os.path.exists(embs_p):
            ids, embs = [], []
            with open(ad_data_path, 'r') as f:
                for line in tqdm(f, desc="Loading ad_data"):
                    aid, _, estr = line.strip().split('\t')
                    ids.append(aid)
                    embs.append(np.fromstring(estr, sep=',', dtype=np.float32))
            ids  = np.array(ids)
            embs = np.stack(embs, axis=0)
            np.save(map_p, ids)
            np.save(embs_p, embs)
        else:
            ids  = np.load(map_p, allow_pickle=True)
            embs = np.load(embs_p, allow_pickle=True)

        # 2) 构造映射及 embedding table
        self.ad2idx = {aid: i+1 for i,aid in enumerate(ids)}  # 1..N
        self.ad2idx['PAD'] = 0
        self.idx2ad  = {idx:aid for aid,idx in self.ad2idx.items()}
        self.emb_dim = embs.shape[1]
        table = np.zeros((len(self.ad2idx), self.emb_dim), dtype=np.float32)
        table[1:] = embs
        self.emb_table = table  # numpy [V, D]

        # 3) 读序列滑窗
        self.maxlen  = maxlen
        self.samples = []
        with open(seq_data_path, 'r') as f:
            for line in tqdm(f, desc="Building samples"):
                _, seq_str = line.strip().split('\t')
                ads = seq_str.split()
                idxs = [self.ad2idx[a] for a in ads if a in self.ad2idx]
                L = len(idxs)
                if L < 2: 
                    continue
                if L > maxlen+1:
                    idxs = idxs[-(maxlen+1):]
                # input seq and target seq aligned
                inp = idxs[:-1]  # [L-1]
                tgt = idxs[1:]   # [L-1]
                self.samples.append((inp, tgt))
        print("Total samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]  # (List[int], List[int])

    def collate_fn(self, batch):
        """
        batch: List of (inp_seq, tgt_seq), both length L_i
        return:
          seqs: [B,T] int64
          tgts: [B,T] int64
          mask:[B,T] float32
        """
        seqs, tgts = zip(*batch)
        B = len(seqs)
        T = max(len(s) for s in seqs)
        seq_pad, tgt_pad, mask = [], [], []
        for s, t in zip(seqs, tgts):
            L = len(s)
            p = T - L
            seq_pad.append([0]*p + s)
            tgt_pad.append([0]*p + t)
            mask.append([0]*p + [1]*L)
        return (
            paddle.to_tensor(seq_pad, dtype='int64'),
            paddle.to_tensor(tgt_pad, dtype='int64'),
            paddle.to_tensor(mask,    dtype='float32')
        )

    def get_embedding_table(self):
        """返回 numpy array [V, D]"""
        return self.emb_table

    def idx2ad(self, idx):
        return self.idx2ad.get(int(idx), 'PAD')