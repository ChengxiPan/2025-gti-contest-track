import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class TwoTowerModel(nn.Layer):
    def __init__(self,
                 ad_emb_table: paddle.Tensor,
                 pad_idx: int = 0,
                 hidden_size: int = 1024):
        """
        ad_emb_table: [num_embs, D]，numpy→tensor 后传进来
        pad_idx:      用于序列 padding 的 index（对应 embedding 全 0）
        hidden_size:  GRU 隐藏维度(D)
        """
        super().__init__()
        num_embs, emb_dim = ad_emb_table.shape

        # 1) Embedding 层，用来把 [B,T] → [B,T,D]，PAD 自动映成全 0
        self.embed = nn.Embedding(
            num_embeddings = num_embs,
            embedding_dim   = emb_dim,
            padding_idx     = pad_idx)
        self.embed.weight.set_value(ad_emb_table)
        self.embed.weight.stop_gradient = True   # 冻结它

        # 2) 序列编码器：GRU
        self.encoder = nn.GRU(input_size=emb_dim,
                              hidden_size=hidden_size,
                              direction='forward')

    def forward(self, input_ids):
        """
        正常前向，只输出 user_vec，用于召回/在线向量化
        input_ids: [B, T] int64
        return: user_vec [B, hidden_size]
        """
        x, _ = self.embed(input_ids), None   # [B,T,D]
        _, h_n = self.encoder(x)             # h_n: [1, B, D]
        user_vec = h_n.squeeze(0)            # [B, D]
        return user_vec

    def nce_loss(self, user_vec, pos_idx, neg_k=100):
        """
        负采样 NCE 损失
        user_vec: [B, D]
        pos_idx:  [B]     正样本广告下标
        neg_k:    每个正例采样多少负例
        returns:  loss, logits_sampled
        """
        B, D = user_vec.shape
        # 1) 采负样本
        #   uniform 采样 [0, num_embs)—包含 pad_idx(0) 也可能被当负样本
        neg_idx = paddle.randint(
            low=0, high=self.embed._num_embeddings,
            shape=[B, neg_k], dtype='int64')

        # 2) 拼正负
        pos_idx = pos_idx.unsqueeze(1)            # [B,1]
        all_idx = paddle.concat([pos_idx, neg_idx], axis=1)  # [B,1+K]

        # 3) 拿到它们的 embedding
        #    [B,1+K,D]
        cand_emb = self.embed(all_idx)

        # 4) 计算点积得分
        #    expand user_vec → [B,1+K,D]
        u = user_vec.unsqueeze(1).expand_as(cand_emb)
        logits = paddle.sum(u * cand_emb, axis=-1)  # [B,1+K]

        # 5) 构造标签：正样本都在位置 0
        labels = paddle.zeros([B], dtype='int64')

        # 6) 交叉熵
        loss = F.cross_entropy(logits, labels)
        return loss, logits

    @paddle.no_grad()
    def recall(self, user_vec, K=100):
        """两塔召回：full-dot + topk"""
        all_emb = self.embed.weight        # [N, D]
        scores = paddle.matmul(user_vec, all_emb.t())  # [B, N]
        topk_scores, topk_idx = paddle.topk(scores, k=K, axis=1)
        return topk_idx, topk_scores

    # rerank 方法同以前，不再赘述