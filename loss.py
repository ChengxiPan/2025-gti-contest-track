import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class CustomContrastiveLoss(nn.Layer):
    def __init__(self):
        super(CustomContrastiveLoss, self).__init__()

    def forward(self, logits, labels, pad_mask, ad_idxs):
        batch_size, seq_len, dim = logits.shape
        logits_flat = logits.reshape([batch_size * seq_len, dim])
        labels_flat = labels.reshape([batch_size * seq_len, dim])
        pad_mask_flat = pad_mask.reshape([batch_size * seq_len])
        ad_idxs_flat = ad_idxs.reshape([batch_size * seq_len])

        # 计算相似度矩阵
        similarity_matrix = paddle.matmul(logits_flat, labels_flat, transpose_y=True)

        # 构建掩码
        mask = paddle.logical_not(pad_mask_flat).astype('float32')
        mask = mask.unsqueeze(1) * mask.unsqueeze(0)
        similarity_matrix = similarity_matrix * mask

        # 计算softmax
        similarity_matrix = F.softmax(similarity_matrix, axis=1)

        # 构建标签矩阵
        label_matrix = (ad_idxs_flat.unsqueeze(1) == ad_idxs_flat.unsqueeze(0)).astype('float32')
        label_matrix = label_matrix * mask

        # 计算损失
        loss = -paddle.log(similarity_matrix + 1e-8) * label_matrix
        loss = loss.sum() / (label_matrix.sum() + 1e-8)
        return loss
