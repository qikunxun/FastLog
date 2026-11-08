import torch
import torch_scatter
from torch.nn import functional as F

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_sum(src, index, dim=-1, dim_size=None):
    index = broadcast(index, src, dim)
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)

def scatter_max(src, index, dim=-1, dim_size=None):
    index = broadcast(index, src, dim)
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return torch_scatter.scatter_max(src, index, dim, out, dim_size)[0]


def block_is_in(arr, targets, block_size=10000000):
    result_list = []
    num_blocks = (arr.size(0) + block_size - 1) // block_size

    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, arr.size(0))
        block = arr[start_idx:end_idx]
        block_result = torch.isin(block, targets)
        block_result = torch.nonzero(block_result)[:, 0]
        result_list.append(block_result + start_idx)

    return torch.cat(result_list)

def log_loss(p_score, label, E, tau_2, one, thr=1e-7):
    one_hot = F.one_hot(label, E).float()
    loss = -torch.sum(
        one_hot * torch.log(torch.maximum(p_score / tau_2, one * thr)),
        dim=-1)
    loss = torch.mean(loss)
    return loss

def log_loss_common(p_score, label, E, tau_2, thr=1e-7):
    i_y = label.long()
    i_x = torch.arange(0, i_y.shape[0]).to(i_y.device)
    i = torch.stack([i_x, i_y], dim=0)
    v = torch.ones_like(label).float()
    one_hot = torch.sparse.FloatTensor(i.long(), v, torch.Size([i_y.shape[0], E])).to_dense()
    loss = -torch.sum(
        one_hot * torch.log(torch.maximum(p_score / tau_2, torch.ones_like(p_score) * thr)),
        dim=-1)
    loss = torch.mean(loss)
    return loss

def log_loss_common_sp(p_score, label, E, tau_2, thr=1e-7):
    i_y = label.long()
    i_x = torch.arange(0, i_y.shape[0]).to(i_y.device)
    i = torch.stack([i_x, i_y], dim=0)
    v = torch.ones_like(label).float()
    one_hot = torch.sparse.FloatTensor(i.long(), v, torch.Size([i_y.shape[0], E]))
    logits = torch.sparse.sum(p_score * one_hot, dim=-1).to_dense()
    loss = -torch.mean(
        torch.log(torch.maximum(logits / tau_2, torch.ones_like(logits) * thr)),
        dim=-1)
    return loss

def vectorized_operation(A, B, target_size, r_size, is_max=False, topk_pruning=100000, weight=None,
                                    use_topk=False, wot_i=False):
    scatter = scatter_sum
    if is_max: scatter = scatter_max

    row_indices, col_indices, r_indices, mask_values, w = B
    non_zero = torch.nonzero(A.sum(1))
    non_zero = torch.unique(non_zero[:, 1])

    non_zero_ori = block_is_in(row_indices, non_zero)
    row_indices_ori = torch.index_select(row_indices, index=non_zero_ori, dim=0)
    col_indices_ori = torch.index_select(col_indices, index=non_zero_ori, dim=0)
    mask_values_ori = torch.index_select(mask_values, index=non_zero_ori, dim=0)
    r_indices_ori = torch.index_select(r_indices, index=non_zero_ori, dim=0)
    if weight is not None:
        C_ori = torch.index_select(weight, index=non_zero_ori.to(weight.device), dim=0).to(A.device)
        C_ori = score_function(C_ori)

    non_zero_inv = block_is_in(col_indices, non_zero)
    row_indices_inv = torch.index_select(col_indices, index=non_zero_inv, dim=0)
    col_indices_inv = torch.index_select(row_indices, index=non_zero_inv, dim=0)
    mask_values_inv = torch.index_select(mask_values, index=non_zero_inv, dim=0)
    r_indices_inv = torch.index_select(r_indices, index=non_zero_inv, dim=0)
    if weight is not None:
        C_inv = torch.index_select(weight, index=non_zero_inv.to(weight.device), dim=0).to(A.device)
        C_inv = score_function(C_inv)

    A_values_ori = torch.index_select(A, dim=-1, index=row_indices_ori)
    if use_topk and A_values_ori.shape[-1] > topk_pruning:
        A_values_ori_topk, A_values_ori_topk_indices = torch.topk(A_values_ori, k=topk_pruning)
        B_values_ori = torch.index_select(w[:, :, :r_size], index=r_indices_ori, dim=2)
        B_values_ori_topk = torch.gather(B_values_ori, index=A_values_ori_topk_indices, dim=-1)
        mask_values_ori_topk = mask_values_ori[A_values_ori_topk_indices]
        result_values_ori = A_values_ori_topk * B_values_ori_topk * mask_values_ori_topk
        if weight is not None:
            C_ori_topk = C_ori.squeeze(dim=-1)[A_values_ori_topk_indices]
            result_values_ori = result_values_ori * C_ori_topk
        col_indices_ori = col_indices_ori[A_values_ori_topk_indices]
        result_ori = scatter(result_values_ori, col_indices_ori.long(), dim=2, dim_size=target_size)
    else:
        B_values_ori = torch.index_select(w[:, :, :r_size], index=r_indices_ori, dim=2)
        result_values_ori = A_values_ori * B_values_ori * mask_values_ori
        if weight is not None:
            result_values_ori = result_values_ori * C_ori.squeeze(dim=-1)
        result_ori = scatter(result_values_ori, col_indices_ori.long(), dim=2, dim_size=target_size)

    A_values_inv = torch.index_select(A, dim=-1, index=row_indices_inv)
    if use_topk and A_values_inv.shape[-1] > topk_pruning:
        A_values_inv_topk, A_values_inv_topk_indices = torch.topk(A_values_inv, k=topk_pruning)
        B_values_inv = torch.index_select(w[:, :, r_size:2 * r_size], index=r_indices_inv, dim=2)
        B_values_inv_topk = torch.gather(B_values_inv, index=A_values_inv_topk_indices, dim=-1)
        mask_values_inv_topk = mask_values_inv[A_values_inv_topk_indices]
        result_values_inv = A_values_inv_topk * B_values_inv_topk * mask_values_inv_topk
        if weight is not None:
            C_inv_topk = C_inv.squeeze(dim=-1)[A_values_inv_topk_indices]
            result_values_inv = result_values_inv * C_inv_topk
        col_indices_inv = col_indices_inv[A_values_inv_topk_indices]
        result_inv = scatter(result_values_inv, col_indices_inv.long(), dim=2, dim_size=target_size)
    else:
        B_values_inv = torch.index_select(w[:, :, r_size:2 * r_size], index=r_indices_inv, dim=2)
        result_values_inv = A_values_inv * B_values_inv * mask_values_inv
        if weight is not None:
            result_values_inv = result_values_inv * C_inv.squeeze(dim=-1)
        result_inv = scatter(result_values_inv, col_indices_inv.long(), dim=2, dim_size=target_size)

    result_ind = None
    if not wot_i:
        result_ind = torch.einsum('ble,bl->ble', A, w[:, :, -1])

    return result_ind, result_ori, result_inv


def vectorized_operation_sp(A, B, E, r_size, is_max=False, topk_pruning=100000, wot_i=False, weight=None, use_topk=False):
    scatter = scatter_sum
    if is_max: scatter = scatter_max

    row_indices_ori, col_indices_ori, r_indices_ori, mask_values_ori, w_all = B
    indices_all, results_all = [], []
    indices_all_inv, results_all_inv = [], []
    indices_all_ind, results_all_ind = [], []
    batch_size = A.shape[0]
    is_first = len(A.shape) == 1
    if not is_first:
        A_batch = torch.unbind(A, dim=0)
    L = w_all.shape[1]
    for i in range(batch_size):
        w = w_all[i].t()
        if is_first:
            A_indices = A[i].unsqueeze(-1)
            A_values = torch.ones_like(A_indices).repeat(1, L)
        else:
            A_ = A_batch[i].coalesce()
            A_indices = A_.indices()[0]
            A_values = A_.values()
        non_zero_ori = A_indices
        non_zero = block_is_in(row_indices_ori, non_zero_ori)
        row_indices = torch.index_select(row_indices_ori, index=non_zero, dim=0)
        col_indices = torch.index_select(col_indices_ori, index=non_zero, dim=0)
        mask_values = torch.index_select(mask_values_ori, index=non_zero, dim=0)
        r_indices = torch.index_select(r_indices_ori, index=non_zero, dim=0)
        if weight is not None:
            C_ori = torch.index_select(weight, index=non_zero.to(weight.device), dim=0).to(A.device)
            C_ori = score_function(C_ori)

        non_zero_inv = block_is_in(col_indices_ori, non_zero_ori)
        row_indices_inv = torch.index_select(col_indices_ori, index=non_zero_inv, dim=0)
        col_indices_inv = torch.index_select(row_indices_ori, index=non_zero_inv, dim=0)
        mask_values_inv = torch.index_select(mask_values_ori, index=non_zero_inv, dim=0)
        r_indices_inv = torch.index_select(r_indices_ori, index=non_zero_inv, dim=0)
        if weight is not None:
            C_inv = torch.index_select(weight, index=non_zero_inv.to(weight.device), dim=0).to(A.device)
            C_inv = score_function(C_inv)

        sorted_indices = torch.searchsorted(non_zero_ori, row_indices)
        A_values_ori = torch.index_select(A_values, dim=0, index=sorted_indices)
        if use_topk:
            k_ = min(topk_pruning, A_values_ori.shape[0])
            A_values_ori_topk, A_values_ori_topk_indices = torch.topk(A_values_ori, k=k_, dim=0)
            B_values_ori = torch.index_select(w[:r_size, :], index=r_indices, dim=0)
            B_values_ori_topk = torch.gather(B_values_ori, index=A_values_ori_topk_indices, dim=0)
            mask_values_ori_topk = mask_values[A_values_ori_topk_indices]
            result_values_ori = A_values_ori_topk * B_values_ori_topk * mask_values_ori_topk
            if weight is not None:
                C_ori_topk = C_ori.squeeze(dim=1)[A_values_ori_topk_indices]
                result_values_ori = result_values_ori * C_ori_topk
            col_indices_ori_topk = col_indices[A_values_ori_topk_indices]
            col_indices_uni = torch.unique(col_indices_ori_topk)
            sorted_indices = torch.searchsorted(col_indices_uni, col_indices_ori_topk)
            result_ori = scatter(result_values_ori, sorted_indices, dim=0, dim_size=col_indices_uni.shape[0])
        else:
            B_values_ori = torch.index_select(w[:r_size, :], index=r_indices, dim=0)
            result_values_ori = A_values_ori * B_values_ori * mask_values.unsqueeze(dim=1)
            if weight is not None:
                result_values_ori = result_values_ori * C_ori.squeeze(dim=1)
            col_indices_uni = torch.unique(col_indices)
            sorted_indices = torch.searchsorted(col_indices_uni, col_indices)
            result_ori = scatter(result_values_ori, sorted_indices, dim=0, dim_size=col_indices_uni.shape[0])

        index = torch.ones_like(col_indices_uni) * i
        index = torch.stack([index, col_indices_uni], dim=0)
        indices_all.append(index)
        results_all.append(result_ori)

        sorted_indices_inv = torch.searchsorted(A_indices, row_indices_inv)
        A_values_inv = torch.index_select(A_values, dim=0, index=sorted_indices_inv)
        if use_topk:
            k_ = min(topk_pruning, A_values_inv.shape[0])
            A_values_inv_topk, A_values_inv_topk_indices = torch.topk(A_values_inv, k=k_, dim=0)
            B_values_inv = torch.index_select(w[r_size: 2 * r_size, :], index=r_indices_inv, dim=0)
            B_values_inv_topk = torch.gather(B_values_inv, index=A_values_inv_topk_indices, dim=0)
            mask_values_inv_topk = mask_values_inv[A_values_inv_topk_indices]
            result_values_inv = A_values_inv_topk * B_values_inv_topk * mask_values_inv_topk
            if weight is not None:
                C_inv_topk = C_inv.squeeze(dim=1)[A_values_inv_topk_indices]
                result_values_inv = result_values_inv * C_inv_topk
            col_indices_inv_topk = col_indices_inv[A_values_inv_topk_indices]
            col_indices_uni_inv = torch.unique(col_indices_inv_topk)
            sorted_indices_inv = torch.searchsorted(col_indices_uni_inv, col_indices_inv_topk)
            result_inv = scatter(result_values_inv, sorted_indices_inv.long(), dim=0,
                                     dim_size=col_indices_uni_inv.shape[0])
        else:
            B_values_inv = torch.index_select(w[r_size: 2 * r_size, :], index=r_indices_inv, dim=0)
            result_values_inv = A_values_inv * B_values_inv * mask_values_inv.unsqueeze(dim=1)
            if weight is not None:
                result_values_inv = result_values_inv * C_inv.squeeze(dim=1)
            col_indices_uni_inv = torch.unique(col_indices_inv)
            sorted_indices_inv = torch.searchsorted(col_indices_uni_inv, col_indices_inv)
            result_inv = scatter(result_values_inv, sorted_indices_inv, dim=0,
                                 dim_size=col_indices_uni_inv.shape[0])

        index = torch.ones_like(col_indices_uni_inv) * i
        index = torch.stack([index, col_indices_uni_inv], dim=0)
        indices_all_inv.append(index)
        results_all_inv.append(result_inv)

        if not wot_i:
            result_ind = A_values * w[-1, :].unsqueeze(dim=0)
            index = torch.ones_like(A_indices) * i
            index = torch.stack([index, A_indices], dim=0)
            indices_all_ind.append(index)
            results_all_ind.append(result_ind)

    i = torch.cat(indices_all, dim=-1)
    v = torch.cat(results_all, dim=0)
    output_ori = torch.sparse.FloatTensor(i.long(), v, torch.Size([batch_size, E, L]))

    i = torch.cat(indices_all_inv, dim=-1)
    v = torch.cat(results_all_inv, dim=0)
    output_inv = torch.sparse.FloatTensor(i.long(), v, torch.Size([batch_size, E, L]))

    if not wot_i:
        i = torch.cat(indices_all_ind, dim=-1)
        v = torch.cat(results_all_ind, dim=0)
        output_ind = torch.sparse.FloatTensor(i.long(), v, torch.Size([batch_size, E, L]))
    else:
        output_ind = None

    return output_ind, output_ori, output_inv


def norm_sp(s):
    batches = torch.unbind(s)
    batches_new = []
    for item in batches:
        s_i = item.coalesce()
        values = s_i.values()
        values = values / values.sum(dim=0, keepdims=True).clamp(1e-7)
        sp_new = torch.sparse.FloatTensor(s_i.indices(), values, s_i.shape)
        batches_new.append(sp_new)
    return torch.stack(batches_new, dim=0)

def sum_sp(s):
    A = s.coalesce()
    shape = A.shape
    indices = A.indices()
    values = A.values().sum(dim=-1)
    return torch.sparse.FloatTensor(indices, values, torch.Size([shape[0], shape[1]]))

def max_sp(s):
    A = s.coalesce()
    shape = A.shape
    indices = A.indices()
    values = A.values().max(dim=-1)[0]
    return torch.sparse.FloatTensor(indices, values, torch.Size([shape[0], shape[1]]))

def score_function(x):
    return 0.5 + 0.5 * torch.sigmoid(x)

def sparse_max(s_i, s_h, s_t):

    s_i = s_i.coalesce()
    s_h = s_h.coalesce()
    s_t = s_t.coalesce()

    indices_list = [s_i.indices(), s_h.indices(), s_t.indices()]
    values_list = [s_i.values(), s_h.values(), s_t.values()]
    
    all_indices = torch.cat(indices_list, dim=1)
    all_values = torch.cat(values_list, dim=0)
    
    batch_size, E, L = s_i.shape
    keys = all_indices[0] * E + all_indices[1]
    
    sorted_keys, sort_idx = torch.sort(keys)
    unique_keys, inverse_indices = torch.unique_consecutive(sorted_keys, return_inverse=True)
    
    sorted_values = all_values[sort_idx]
    
    max_values = torch.zeros((len(unique_keys), L), 
                           dtype=s_i.values().dtype, 
                           device=s_i.values().device)
    
    max_values.scatter_reduce_(0, 
                             inverse_indices.unsqueeze(1).expand(-1, L), 
                             sorted_values, 
                             reduce='amax', 
                             include_self=False)
    
    batch_indices = unique_keys // E
    E_indices = unique_keys % E
    result_indices = torch.stack([batch_indices, E_indices])
    
    return torch.sparse_coo_tensor(result_indices, max_values, s_i.shape)