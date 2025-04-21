import paddle
import paddle.nn as nn

class SASRec(nn.Layer):
    """
    只做序列编码，输出 [B,T,H] 特征。
    """
    def __init__(self, num_items, maxlen, hidden, heads, blocks, dropout):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, hidden, padding_idx=0)
        self.pos_emb  = nn.Embedding(maxlen+1, hidden, padding_idx=0)
        self.dropout  = nn.Dropout(dropout)
        self.layers   = nn.LayerList([
            nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=heads,
                dim_feedforward=hidden*4,
                dropout=dropout,
                activation='relu')
            for _ in range(blocks)
        ])
        self.layer_norm = nn.LayerNorm(hidden)

    def encode(self, seqs, mask):
        """
        seqs: [B, T], mask: [B, T]
        return: [B, T, H]
        """
        B, T = seqs.shape
        x = self.item_emb(seqs)  # [B,T,H]
        ar = paddle.arange(1, T+1, dtype='int64').unsqueeze(0).expand([B, T])
        pos_ids = (ar * mask.astype('int64')).astype('int64')
        x = x + self.pos_emb(pos_ids)
        x = self.dropout(x)

        x = x.transpose([1,0,2])  # [T,B,H]
        for layer in self.layers:
            x = layer(x)           # 全注意，无因果 mask
        x = x.transpose([1,0,2])  # [B,T,H]
        return self.layer_norm(x) # [B,T,H]