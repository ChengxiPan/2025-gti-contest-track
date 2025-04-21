import paddle
import numpy as np
from tqdm import tqdm
from model import TwoTowerModel
import os
import pickle

checkpoint_dir = 'checkpoint'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

model_path = os.path.join(checkpoint_dir, 'two_tower_model.pdparams')


# === 加载 mapper ===
with open("saved_mapper.pkl", "rb") as f:
    mapper = pickle.load(f)
num_ads = len(mapper)

# === 加载模型 ===
model = TwoTowerModel(hidden_size=1024, num_ads=num_ads)
model.set_state_dict(paddle.load(model_path))
model.eval()


# === 推理逻辑 ===
sample_submission_path = 'sample_submission.txt'
read_path = 'data/sequence_data'

results = []

with open(read_path, 'r') as f:
    for line in tqdm(f, desc="Predicting"):
        user_id, ad_seq = line.strip().split('\t')
        ad_ids = ad_seq.strip().split()

        if not ad_ids:
            continue

        # 只使用最后 max_len 个（模型训练时也有限长序列）
        max_len = 20
        input_ids = ad_ids[-max_len:]

        # === embedding lookup 并 padding ===
        input_embs = []
        for aid in input_ids:
            emb = mapper.get_embedding(aid)
            if emb is not None:
                input_embs.append(emb)

        if not input_embs:
            continue  # 跳过无法映射的用户

        pad_len = max_len - len(input_embs)
        input_embs = [mapper.pad_emb] * pad_len + input_embs  # padding to left
        input_array = np.stack(input_embs)[np.newaxis, :]     # [1, T, D]

        input_tensor = paddle.to_tensor(input_array, dtype='float32')  # [1, T, D]

        # === 模型预测 ===
        with paddle.no_grad():
            output = model(input_tensor, paddle.to_tensor(mapper.get_all_embeddings()))  # [1, num_ads]
            prediction_idx = paddle.argmax(output, axis=1).numpy()[0]
            prediction_ad_id = mapper.index2id[prediction_idx]

        results.append(f"{user_id}\t{prediction_ad_id}")

# === 写入预测结果 ===
with open(sample_submission_path, 'w') as out_file:
    out_file.write('\n'.join(results))
