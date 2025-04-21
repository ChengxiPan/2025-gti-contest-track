import paddle
from paddle.io import DataLoader
from paddle.optimizer import Adam
from tqdm import tqdm
import pickle
import random

from utils import AdEmbeddingMapper, AdSequenceDataset, collate_fn
from model import TwoTowerModel

paddle.set_device('gpu')
random.seed(42)

# 1) 加载 Dataset & Mapper
with open("saved_ds.pkl", "rb") as f:
    ds = pickle.load(f)
with open("saved_mapper.pkl", "rb") as f:
    mapper = pickle.load(f)

dl = DataLoader(
    ds,
    batch_size=256,
    shuffle=True,
    num_workers=4,    # 多进程加载
    drop_last=True,
    collate_fn=collate_fn)

# 2) 准备 ad embedding table
emb_table = paddle.to_tensor(
    mapper.get_emb_table(), dtype='float32')

# 3) 初始化模型/优化器
model = TwoTowerModel(
    ad_emb_table=emb_table,
    pad_idx=mapper.pad_idx,
    hidden_size=emb_table.shape[1])
optimizer = Adam(parameters=model.parameters(), learning_rate=1e-3)

# 4) 训练：召回塔只用 NCE loss
model.train()                # 切到训练模式
for epoch in range(10):
    pbar = tqdm(dl, desc=f"Epoch {epoch}", dynamic_ncols=True)
    for input_ids, mask, target in pbar:
        optimizer.clear_grad()                   # 【1】每个 batch 开始前清一次梯度

        user_vec = model(input_ids)              # [B, D]
        loss, _ = model.nce_loss(user_vec,       # 负采样 NCE
                                  pos_idx=target,
                                  neg_k=200)

        loss.backward()                          # 反向
        optimizer.step()                         # 更新

        # 取一个 Python float，更轻量
        pbar.set_postfix({"loss": float(loss)})

    # epoch 结束时，你可以再打印一下
    print(f"Epoch {epoch} final loss: {float(loss):.4f}")

    # 5) 保存模型
    paddle.save(model.state_dict(), f"checkpoint/two_tower_nce/epoch-{epoch}.pdparams")