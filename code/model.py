import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class TwoTowerModel(nn.Layer):
    def __init__(self, emb_table, pad_idx=0, hidden_size=1024):
        super().__init__()
        num_ads, emb_dim = emb_table.shape
        self.embedding = nn.Embedding(
            num_embeddings=num_ads,
            embedding_dim=emb_dim,
            padding_idx=pad_idx)
        self.embedding.weight.set_value(emb_table)
        self.embedding.weight.stop_gradient = True

        self.encoder = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_size,
            direction='forward')
        # bias for sampled softmax
        self.bias = self.create_parameter([num_ads], default_initializer=nn.initializer.Constant(0.))

    def forward(self, input_ids):
        """
        input_ids: [B, T]
        Returns: user_vec [B, hidden_size]
        """
        x = self.embedding(input_ids)     # [B, T, D]
        _, h = self.encoder(x)            # [1, B, H]
        return h.squeeze(0)               # [B, H]

    def sampled_softmax_loss(self, user_vec, labels, num_sampled=100):
        """
        user_vec: [B, H], labels: [B]
        """
        loss, _ = paddle.fluid.layers.sampled_softmax_with_cross_entropy(
            weight=self.embedding.weight,  # [N, D]
            bias=self.bias,                # [N]
            label=labels.unsqueeze(1),     # [B,1]
            input=user_vec,                # [B,D]
            num_true=1,
            sampled_count=num_sampled,
            unique=True,
            range_max=self.embedding.num_embeddings)
        return paddle.mean(loss)

    @paddle.no_grad()
    def recall(self, user_vec, K=10):
        """
        user_vec: [B, H]
        """
        scores = paddle.matmul(user_vec, self.embedding.weight, transpose_y=True)  # [B,N]
        topk_scores, topk_idx = paddle.topk(scores, k=K, axis=1)
        return topk_idx, topk_scores