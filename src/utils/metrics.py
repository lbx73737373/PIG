import numpy as np
import torch
import ipdb

def cal_cossim(feats1, feats2):
    sim_matrix = np.dot(feats1, feats2.T)
    return sim_matrix

def cal_cossim_multi(text_feats, text_all_feats, text_input_mask, video_feats, video_all_feats, mode):
    # text_feats: (B, D), video_feats: (B, D)
    # text_all_feats: (B, N, D), video_all_feats: (B, M, D)
    # text_input_mask: (B, N)
    if isinstance(text_feats, np.ndarray):
        text_feats = torch.from_numpy(text_feats).float()
        text_all_feats = torch.from_numpy(text_all_feats).float()
        video_feats = torch.from_numpy(video_feats).float()
        video_all_feats = torch.from_numpy(video_all_feats).float()
        text_input_mask = torch.from_numpy(text_input_mask).bool()
        B, N, D = text_all_feats.size()
        M = video_all_feats.size(1)
    if mode == 'word-video':
        # sim_matrix: (B, N, B)
        sim_matrix = torch.matmul(text_all_feats, video_feats.T)
        text_input_mask = text_input_mask.unsqueeze(-1).expand_as(sim_matrix).bool()
        sim_matrix[~text_input_mask] = -float('inf')
        sim_matrix_maxpool, max_index = torch.max(sim_matrix, dim=1)

        sim_matrix = sim_matrix.numpy()
        sim_matrix_maxpool = sim_matrix_maxpool.numpy()
        max_index = max_index.numpy()
        diag_sim_matrix = None
    elif mode == 'word-frame':
        # sim_matrix: (B, N, M, B)
        # B, N, M
        text_all_feats = text_all_feats.reshape(-1, D)
        video_all_feats = video_all_feats.reshape(-1, D)
        # (B*N, D) * (B*M, D).T = (B*N, B*M)
        sim_matrix = torch.matmul(text_all_feats, video_all_feats.T)
        sim_matrix = sim_matrix.reshape(B, N, B, M)
        sim_matrix = sim_matrix.permute(0, 1, 3, 2)
        sim_matrix = sim_matrix.reshape(sim_matrix.size(0), -1, sim_matrix.size(-1))
        sim_matrix_maxpool, max_index = torch.max(sim_matrix, dim=1)

        sim_matrix = sim_matrix.numpy()
        sim_matrix_maxpool = sim_matrix_maxpool.numpy()
        max_index = max_index.numpy()
        diag_sim_matrix = None
    elif mode == 'word-video_only_positive':
        # (B, 1, D)
        video_feats = video_feats.unsqueeze(1)
        # (B, N, D) * (B, D, 1) = (B, N)
        diag_sim_matrix = torch.bmm(text_all_feats, video_feats.transpose(1, 2)).squeeze(-1)
        text_input_mask = text_input_mask.bool()
        diag_sim_matrix[~text_input_mask] = -float('inf')
        diag_sim_matrix_maxpool, max_index = torch.max(diag_sim_matrix, dim=1)
        sim_matrix = torch.matmul(text_feats, video_feats.squeeze(1).transpose(0, 1))
        sim_matrix[torch.arange(B), torch.arange(B)] = diag_sim_matrix_maxpool
        sim_matrix_maxpool = sim_matrix

        sim_matrix = sim_matrix.numpy()
        sim_matrix_maxpool = sim_matrix_maxpool.numpy()
        max_index = max_index.numpy()
        diag_sim_matrix = diag_sim_matrix.numpy()
    elif mode == 'word-video_top3':
        # sim_matrix: (B, N, B)
        sim_matrix = torch.matmul(text_all_feats, video_feats.T)
        text_input_mask = text_input_mask.unsqueeze(-1).expand_as(sim_matrix).bool()
        sim_matrix[~text_input_mask] = -float('inf')

        # 获取 top-3 的 token 相似度，然后对 top-3 的值求平均
        # sim_matrix 的维度为 (B, N, B)，在维度1（即 N 上）进行 topk
        top_val, top_idx = torch.topk(sim_matrix, 3, dim=1)  # top_val: (B,3,B)
        sim_matrix_top3_mean = top_val.mean(dim=1)  # (B,B) 对 top-3 的值求均值

        sim_matrix = sim_matrix.numpy()
        sim_matrix_maxpool = sim_matrix_top3_mean.numpy()  # 这里将maxpool替换为top-3均值
        max_index = top_idx[:, 0, :].numpy()  # top_idx 的第一个是最大值对应的位置，也可以根据需求返回

        diag_sim_matrix = None

    elif mode == 'word-patch':
        # sim_matrix: (B, N, M, B)
        # B, N, M
        text_all_feats = text_all_feats.reshape(-1, D)
        video_all_feats = video_all_feats.reshape(-1, D)
        # (B*N, D) * (B*M, D).T = (B*N, B*M)
        sim_matrix = torch.matmul(text_all_feats, video_all_feats.T)
        sim_matrix = sim_matrix.reshape(B, N, B, M)
        sim_matrix = sim_matrix.permute(0, 1, 3, 2)
        sim_matrix = sim_matrix.reshape(sim_matrix.size(0), -1, sim_matrix.size(-1))
        sim_matrix_maxpool, max_index = torch.max(sim_matrix, dim=1)

        sim_matrix = sim_matrix.numpy()
        sim_matrix_maxpool = sim_matrix_maxpool.numpy()
        max_index = max_index.numpy()
        diag_sim_matrix = None


    else:
        raise NotImplementedError

    return {
        'sim_matrix': sim_matrix,
        'sim_matrix_maxpool': sim_matrix_maxpool,
        'max_index': max_index,
        'diag_sim_matrix': diag_sim_matrix,
    }



def np_softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """
    # make X at least 2d
    y = np.atleast_2d(X)
    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    # multiply y against the theta parameter, 
    y = y * float(theta)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()
    return p

def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    r1 = float(np.sum(ind == 0))  / len(ind)
    r5 = float(np.sum(ind < 5))  / len(ind)
    r10 = float(np.sum(ind < 10))  / len(ind)
    medr = np.median(ind) + 1
    meanr  = np.mean(ind) + 1
    return r1, r5, r10, medr, meanr

def compute_metrics_zero_division(x):
    epsilon = np.finfo(float).eps
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    r1 = float(np.sum(ind == 0))  / (len(ind) + epsilon)
    r5 = float(np.sum(ind < 5))  / (len(ind) + epsilon)
    r10 = float(np.sum(ind < 10))  / (len(ind) + epsilon)
    medr = np.median(ind) + 1
    meanr  = np.mean(ind) + 1
    return r1, r5, r10, medr, meanr

def compute_metrics_multi(x, t2v_labels_list):
    sx = np.sort(-x, axis=1)
    t2v_labels_list = np.array(t2v_labels_list)
    arg = np.arange(x.shape[0])
    d = -x[arg, t2v_labels_list]
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    r1 = float(np.sum(ind == 0))  / len(ind)
    r5 = float(np.sum(ind < 5))  / len(ind)
    r10 = float(np.sum(ind < 10))  / len(ind)
    medr = np.median(ind) + 1
    meanr  = np.mean(ind) + 1
    return r1, r5, r10, medr, meanr


if __name__ == '__main__':

    sim_matrix = np.random.random((5,5))



