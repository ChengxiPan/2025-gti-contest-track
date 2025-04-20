import paddle
from paddle.io import Dataset, DataLoader, BatchSampler, DistributedBatchSampler
import numpy as np
import os
import time
from datetime import date
from tqdm import tqdm
from utils import *
from model import *

# 配置文件
class Args():
    def __init__(self):
        self.dataset_dir = "data" # root directory containing the datasets
        self.unitid_file = "unitid.txt"
        self.train_file = "1w_train.txt"
        self.test_file = "test.txt"
        self.test_gt_file = "test_gt.txt"
        self.batch_size = 32
        self.lr = 0.0001
        self.maxlen = 200
        self.hidden_units = 1024
        self.emb_dim = 1024
        self.num_blocks = 2
        self.num_epochs = 3
        self.num_heads = 1
        self.dropout_rate = 0.2
        self.device = "gpu"
        self.inference_only = False
        # self.state_dict_path = "2025_03_27/SASRec.epoch=3.lr=0.0001.layer=2.head=1.hidden=1024.maxlen=200.pth"
        self.state_dict_path = None

args = Args()
with open(os.path.join(args.dataset_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

model = SASRec(args).to(args.device) # no ReLU activation in original SASRec implementation?

# 定义一个 XavierNormal 初始化器
xavier_normal_init = paddle.nn.initializer.XavierNormal()

for name, param in model.named_parameters():
    try:
        xavier_normal_init(param, param.block)
    except:
        print(f"{name} xaiver 初始化失败")
        pass  # 忽略初始化失败的层

model.train() # enable model training

epoch_start_idx = 1
if args.state_dict_path is not None:
    try:
        model.set_dict(paddle.load(args.state_dict_path))
    except:  
        print('failed loading state_dicts, pls check file path: ')

print("开始加载训练数据...")
dataset = TrainDataset(args)
dataloader =  paddle.io.DataLoader(dataset, batch_size=args.batch_size,collate_fn=dataset.collate_fn)
print("数据加载完成")

criterion = CustomContrastiveLoss()
lr_scheduler = paddle.optimizer.lr.ExponentialDecay(
    learning_rate=args.lr,
    gamma=0.96,
    last_epoch=-1,
    verbose=True
)
adam_optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=lr_scheduler, beta1=0.9, beta2=0.98)
best_val_ndcg, best_val_hr = 0.0, 0.0
best_test_ndcg, best_test_hr = 0.0, 0.0
T = 0.0
t0 = time.time()
step = 0
accumulated_step = 0
print("开始训练")
for epoch in range(epoch_start_idx, args.num_epochs + 1):
    if args.inference_only: break # just to decrease identition
    for padded_embeddings, padded_pos_emb, padded_neg_emb, pad_mask, ad_ids in tqdm(dataloader): 
        logits = model(padded_embeddings, pad_mask, padded_pos_emb, padded_neg_emb)
        loss = criterion(logits,padded_pos_emb,pad_mask,ad_ids)
        loss.backward()
        adam_optimizer.step()
        print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
        step += 1
    lr_scheduler.step()
    # 打印当前学习率
    print(f'Epoch {epoch}, Current learning rate: {adam_optimizer.get_lr()}')
    today = date.today()
    day = today.strftime("%Y_%m_%d") # 2023_10_05
    folder = "/home/aistudio" + "/" + day
    if not os.path.exists(folder):
        os.makedirs(folder)
    fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
    fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
    paddle.save(model.state_dict(), os.path.join(folder, fname))
    t0 = time.time()
    model.train()
print("Done") 