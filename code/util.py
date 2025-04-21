# util.py
import os
import numpy as np
import paddle
from paddle.io import Dataset
from tqdm import tqdm

class AdSequenceDataset(Dataset):
    """
    两塔 NCE 训练用 Dataset：
      1) 缓存 ad_data 到 ad_ids.npy/ad_embs.npy
      2) 构造 ad2idx/idx2ad 及 emb_table（一次性 vectorized）
      3) 读 sequence_data，滑窗生成 (input_seq, target) 样本
    """
    def __init__(self, ad_data_path, seq_data_path, max_seq_len=20):
        # 1) 加载或缓存广告 embedding
        map_path  = "ad_ids.npy"
        embs_path = "ad_embs.npy"
        if not os.path.exists(map_path) or not os.path.exists(embs_path):
            ids, embs = [], []
            with open(ad_data_path, 'r') as f:
                for line in tqdm(f, desc="Loading ad_data"):
                    aid, _, emb_str = line.strip().split('\t')
                    ids.append(aid)
                    embs.append(np.fromstring(emb_str, dtype=np.float32, sep=','))
            ids  = np.array(ids)
            embs = np.stack(embs, axis=0)  # [N, D]
            np.save(map_path,  ids)
            np.save(embs_path, embs)
        else:
            print("Loading cached ad_data...")
            ids  = np.load(map_path, allow_pickle=True)
            embs = np.load(embs_path, allow_pickle=True)
        
        print(f"Ad IDs: {len(ids)}")
        print(f"Ad Embeddings: {embs.shape}")

        # 2) 构造映射和 embedding table
        # ad2idx: 给每个广告一个唯一 index，1..N，0 为 PAD
        self.ad2idx = {aid: i+1 for i, aid in enumerate(ids)}
        self.ad2idx['PAD'] = 0
        # idx2ad: 反向映射
        self.idx2ad  = {idx: aid for aid, idx in self.ad2idx.items()}
        self.emb_dim = embs.shape[1]

        # 一次性构造 [N+1, D] embedding table，table[0] 全 0
        Np1 = len(ids) + 1
        table = np.zeros((Np1, self.emb_dim), dtype=np.float32)
        table[1:] = embs   # embs[i] 对应 table[i+1]
        self.emb_table = table

        # 3) 读取序列数据并滑窗
        print(f"Max sequence length: {max_seq_len}")
        self.max_seq_len = max_seq_len
        self.samples = []  # List of (List[int], int)
        with open(seq_data_path, 'r') as f:
            for line in tqdm(f, desc="Building samples"):
                _, seq_str = line.strip().split('\t')
                ads = seq_str.split()
                # 转为 idx（过滤未知）
                idxs = [self.ad2idx[a] for a in ads if a in self.ad2idx]
                L = len(idxs)
                for i in range(1, L):
                    start = max(0, i - self.max_seq_len)
                    inp = idxs[start:i]
                    tgt = idxs[i]
                    self.samples.append((inp, tgt))

        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 返回 (input_seq_idxs, target_idx)
        return self.samples[idx]

    def collate_fn(self, batch):
        """
        batch: List of (List[int], int)
        return:
          input_ids: [B, T] int64
          target:    [B]    int64
        """
        seqs, tgts = zip(*batch)
        B = len(seqs)
        T = max(len(s) for s in seqs)
        padded = []
        for s in seqs:
            pad = T - len(s)
            padded.append([0]*pad + s)
        return (
            paddle.to_tensor(padded, dtype='int64'),
            paddle.to_tensor(tgts,   dtype='int64')
        )

    def get_embedding_table(self):
        """
        返回 numpy array [N+1, D]
        """
        return self.emb_table

    def idx2ad(self, idx):
        """
        将 idx (int) 转回原始 ad_id (str)
        """
        return self.idx2ad.get(int(idx), 'PAD')