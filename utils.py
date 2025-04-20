from paddle.io import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm

class AdsDataset(Dataset):
    def __init__(self, ad_data_path, sequence_data_path, max_seq_len=20, use_emb=True):
        super(AdsDataset, self).__init__()
        self.max_seq_len = max_seq_len
        self.use_emb = use_emb

        # load ad data: 
        ## ad_id -> content list and ad_id -> embedding
        ### key: ad_id, value: content list or embedding
        self.ad_content, self.ad_emb = self._load_ad_data(ad_data_path)

        # load sequence data:
        ## user_id -> ad sequence
        ### [(ad_sequence_ids, target_ad_id), ...]
        self.samples = self._load_sequence_data(sequence_data_path)

    def _load_ad_data(self, ad_data_path):
        ad_content = {}  # ad_id -> content list
        ad_emb = {}      # ad_id -> np.array
        with open(ad_data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue  # skip invalid lines
                ad_id, content_str, emb_str = parts
                ad_id = ad_id.strip() # exg: 123456
                content = list(map(int, content_str.split(','))) # exg: [1, 2, 3, 4]
                emb = np.array(list(map(float, emb_str.split(','))), dtype=np.float32) # exg: [0.1, 0.2, 0.3, 0.4]
                ad_content[ad_id] = content
                ad_emb[ad_id] = emb
        return ad_content, ad_emb

    def _load_sequence_data(self, sequence_data_path):
        samples = []
        with open(sequence_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                user_id, ad_seq = line.strip().split('\t')
                ad_ids = ad_seq.strip().split()
                if len(ad_ids) < 2:
                    continue
                for i in range(1, len(ad_ids)):
                    input_seq = ad_ids[max(0, i - self.max_seq_len):i] # take the last max_seq_len elements
                    target = ad_ids[i] # the next ad to predict
                    samples.append((input_seq, target)) # append the input sequence and target
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, target_id = self.samples[idx]

        if self.use_emb:
            input_embs = [self.ad_emb[aid] for aid in input_ids if aid in self.ad_emb]
            target_emb = self.ad_emb[target_id]
            # padding
            pad_len = self.max_seq_len - len(input_embs)
            if pad_len > 0:
                pad_emb = np.zeros_like(target_emb)
                input_embs = [pad_emb] * pad_len + input_embs
            input_tensor = np.stack(input_embs)
            return input_tensor, target_emb
        else:
            # 输入是 ad content ids（tokens）
            input_tokens = [self.ad_content[aid] for aid in input_ids if aid in self.ad_content]
            target_tokens = self.ad_content[target_id]
            # padding
            pad_token = [0]
            input_tokens = [[0]*10]*(self.max_seq_len - len(input_tokens)) + input_tokens
            return input_tokens, target_tokens


if __name__ == "__main__":
    dataset = AdsDataset(
    ad_data_path="./data/ad_data",
    sequence_data_path="./data/sequence_data",
    max_seq_len=10,
    use_emb=True  # or False for token mode
)

    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # for batch in dataloader:
    #     x, y = batch
    #     print(x.shape, y.shape)
    #     break
