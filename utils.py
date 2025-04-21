import numpy as np
import random
from paddle.io import Dataset
from tqdm import tqdm
import pickle
import paddle
from paddle.io import DataLoader

class AdEmbeddingMapper:
    """
    负责
    1) 加载 ad_id -> emb 映射
    2) 构造 ad_id -> index (1…N)，0 留给 PAD
    3) 输出 embedding table [N+1, D]，index=0 对应全 0 向量
    """
    def __init__(self, ad_data_path):
        self.ad_emb = {}
        with open(ad_data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading ad embeddings"):
                parts = line.strip().split('\t')
                if len(parts)!=3: continue
                aid, _, emb_str = parts
                emb = np.array(list(map(float, emb_str.split(','))), dtype=np.float32)
                self.ad_emb[aid] = emb
        # 建 index，从1开始
        self.id2idx = {}
        self.idx2id = {}
        for i, aid in enumerate(self.ad_emb.keys(), start=1):
            self.id2idx[aid] = i
            self.idx2id[i] = aid
        self.pad_idx = 0
        self.emb_dim = next(iter(self.ad_emb.values())).shape[0]

    def __len__(self):
        return len(self.ad_emb) + 1  # 多一个 pad

    def get_emb_table(self):
        N = len(self.ad_emb)
        table = np.zeros((N+1, self.emb_dim), dtype=np.float32)
        for aid, idx in self.id2idx.items():
            table[idx] = self.ad_emb[aid]
        return table

class AdSequenceDataset(Dataset):
    """
    每个样本返回：
      input_seq: List[int]  (每个 int 是在 mapper.id2idx 中的 index)
      target:    int
    """
    def __init__(self,
                 sequence_data_path,
                 mapper: AdEmbeddingMapper,
                 max_seq_len=20,
                 max_samples_per_user=10):
        super().__init__()
        self.samples = []
        self.mapper = mapper
        self.max_seq = max_seq_len
        self.max_samp = max_samples_per_user

        with open(sequence_data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading sequences"):
                user, seq = line.strip().split('\t')
                ad_ids = seq.split()
                # 截断
                if len(ad_ids) > self.max_seq+1:
                    ad_ids = ad_ids[-(self.max_seq+1):]
                # 拆成多对 (input, target)
                cand = []
                for i in range(1, len(ad_ids)):
                    inp = []
                    for aid in ad_ids[:i]:
                        idx = self.mapper.id2idx.get(aid)
                        if idx is not None:
                            inp.append(idx)
                    tgt = self.mapper.id2idx.get(ad_ids[i], None)
                    if inp and tgt is not None:
                        cand.append((inp, tgt))
                if len(cand)>self.max_samp:
                    cand = random.sample(cand, self.max_samp)
                self.samples.extend(cand)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    """
    batch: List of (List[int], int)
    返回：
      input_ids: [B, T]  int64，前 pad 到 max T
      mask:      [B, T]  int64，0=pad,1=real
      target:    [B]     int64
    """
    seqs, tgts = zip(*batch)
    B = len(seqs)
    maxL = max(len(s) for s in seqs)
    padded = []
    masks  = []
    for s in seqs:
        padlen = maxL - len(s)
        padded.append([0]*padlen + s)
        masks.append([0]*padlen + [1]*len(s))
    return (paddle.to_tensor(padded, dtype='int64'),
            paddle.to_tensor(masks,  dtype='float32'),
            paddle.to_tensor(tgts,   dtype='int64'))


    
if __name__ == "__main__":
    ad_data_path = 'data/ad_data'
    sequence_data_path = 'data/sequence_data'
    
    # 1) 准备 Mapper + Dataset + Dataloader
    mapper = AdEmbeddingMapper('data/ad_data')
    emb_table = mapper.get_emb_table()  # numpy [N+1, D]
    emb_tensor = paddle.to_tensor(emb_table, dtype='float32')

    ds = AdSequenceDataset('data/sequence_data', mapper, max_seq_len=20)
    dl = DataLoader(ds, batch_size=256, shuffle=True, drop_last=True,
                    collate_fn=collate_fn)
    
    # 2) 将数据集保存到文件
    with open("saved_ds.pkl", "wb") as f:
        pickle.dump(ds, f)
    with open("saved_mapper.pkl", "wb") as f:
        pickle.dump(mapper, f)
        
    # 3) 从文件中加载数据集
    with open("saved_ds.pkl", "rb") as f:
        loaded_ds = pickle.load(f)
    with open("saved_mapper.pkl", "rb") as f:
        loaded_mapper = pickle.load(f)
    
    dl = DataLoader(loaded_ds, batch_size=256, shuffle=True, drop_last=True,
                    collate_fn=collate_fn)
    
    for batch in dl:
        input_ids, mask, target = batch
        print("Input IDs:", input_ids)
        print("Mask:", mask)
        print("Target:", target)
        break