import torch
import torch_scatter
from torch.nn import functional as F

def activation(x, one):
    return torch.minimum(x, one.to(x.device))

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
    # one_hot = F.one_hot(label, E).float()
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
    # one_hot = F.one_hot(label, E).float()
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

def sparse_matrix_multiply(A, B, target_size, r_size, tau_1, is_training=False, dropout=None,
                           is_max=False, weight=None, wot_i=False):
    scatter = scatter_sum
    if is_max: scatter = scatter_max

    row_indices, col_indices, r_indices, mask_values, w = B

    non_zero = torch.nonzero(A)
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

    w = torch.softmax(w / tau_1, dim=-1)

    A_values_ori = torch.index_select(A, dim=1, index=row_indices_ori)
    B_values_ori = torch.index_select(w[:, :, :r_size], index=r_indices_ori, dim=2)
    result_values_ori = torch.einsum('bm,blm->blm', A_values_ori, B_values_ori) * mask_values_ori.unsqueeze(0).unsqueeze(0)
    if weight is not None: result_values_ori = result_values_ori * C_ori.t().unsqueeze(dim=0)
    # if is_training: result_values_ori = dropout(result_values_ori)
    result_ori = scatter(result_values_ori, col_indices_ori.long(), dim=2, dim_size=target_size)

    A_values_inv = torch.index_select(A, dim=1, index=row_indices_inv)
    B_values_inv = torch.index_select(w[:, :, r_size:2 * r_size], index=r_indices_inv, dim=2)
    result_values_inv = torch.einsum('bm,blm->blm', A_values_inv, B_values_inv) * mask_values_inv.unsqueeze(0).unsqueeze(0)
    if weight is not None: result_values_inv = result_values_inv * C_inv.t().unsqueeze(dim=0)
    # if is_training: result_values_inv = dropout(result_values_inv)
    result_inv = scatter(result_values_inv, col_indices_inv.long(), dim=2, dim_size=target_size)

    result_ind = None
    if not wot_i:
        result_ind = torch.einsum('bm,bl->blm', A, w[:, :, -1])
    return result_ind, result_ori, result_inv

def sparse_matrix_multiply_L_sample(A, B, target_size, r_size, tau_1, is_training=False,
                                    dropout=None, is_max=False, top_k=1000, topk_pruning=100000, weight=None,
                                    use_topk=False, wot_i=False):
    scatter = scatter_sum
    if is_max: scatter = scatter_max

    row_indices, col_indices, r_indices, mask_values, w = B

    non_zero = torch.nonzero(A.sum(1))
    non_zero = torch.unique(non_zero[:, 1])
    if use_topk:
        k_ = min(top_k, A.shape[-1])
        topk = torch.topk(A.sum(1), k=k_)[1]
        topk = torch.unique(topk.view(-1))
        if topk.shape[0] < non_zero.shape[0]:
            non_zero = topk

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

    w = torch.softmax(w / tau_1, dim=-1)

    A_values_ori = torch.index_select(A, dim=-1, index=row_indices_ori)
    if use_topk:
        k_ = min(topk_pruning, A_values_ori.shape[-1])
        A_values_ori_topk, A_values_ori_topk_indices = torch.topk(A_values_ori, k=k_)
        B_values_ori = torch.index_select(w[:, :, :r_size], index=r_indices_ori, dim=2)
        B_values_ori_topk = torch.gather(B_values_ori, index=A_values_ori_topk_indices, dim=-1)
        mask_values_ori_topk = mask_values_ori[A_values_ori_topk_indices]
        result_values_ori = A_values_ori_topk * B_values_ori_topk * mask_values_ori_topk
        if weight is not None:
            C_ori_topk = C_ori.squeeze(dim=-1)[A_values_ori_topk_indices]
            result_values_ori = result_values_ori * C_ori_topk
        # if is_training: result_values_ori = dropout(result_values_ori)
        col_indices_ori = col_indices_ori[A_values_ori_topk_indices]
        result_ori = scatter(result_values_ori, col_indices_ori.long(), dim=2, dim_size=target_size)
    else:
        B_values_ori = torch.index_select(w[:, :, :r_size], index=r_indices_ori, dim=2)
        result_values_ori = A_values_ori * B_values_ori * mask_values_ori
        if weight is not None:
            result_values_ori = result_values_ori * C_ori.squeeze(dim=-1)
        # if is_training: result_values_ori = dropout(result_values_ori)
        result_ori = scatter(result_values_ori, col_indices_ori.long(), dim=2, dim_size=target_size)

    A_values_inv = torch.index_select(A, dim=-1, index=row_indices_inv)
    if use_topk:
        k_ = min(topk_pruning, A_values_inv.shape[-1])
        A_values_inv_topk, A_values_inv_topk_indices = torch.topk(A_values_inv, k=k_)
        B_values_inv = torch.index_select(w[:, :, r_size:2 * r_size], index=r_indices_inv, dim=2)
        B_values_inv_topk = torch.gather(B_values_inv, index=A_values_inv_topk_indices, dim=-1)
        mask_values_inv_topk = mask_values_inv[A_values_inv_topk_indices]
        result_values_inv = A_values_inv_topk * B_values_inv_topk * mask_values_inv_topk
        if weight is not None:
            C_inv_topk = C_inv.squeeze(dim=-1)[A_values_inv_topk_indices]
            result_values_inv = result_values_inv * C_inv_topk
        # if is_training: result_values_inv = dropout(result_values_inv)
        col_indices_inv = col_indices_inv[A_values_inv_topk_indices]
        result_inv = scatter(result_values_inv, col_indices_inv.long(), dim=2, dim_size=target_size)
    else:
        B_values_inv = torch.index_select(w[:, :, r_size:2 * r_size], index=r_indices_inv, dim=2)
        result_values_inv = A_values_inv * B_values_inv * mask_values_inv
        if weight is not None:
            result_values_inv = result_values_inv * C_inv.squeeze(dim=-1)
        # if is_training: result_values_inv = dropout(result_values_inv)
        result_inv = scatter(result_values_inv, col_indices_inv.long(), dim=2, dim_size=target_size)

    result_ind = None
    if not wot_i:
        result_ind = torch.einsum('ble,bl->ble', A, w[:, :, -1])

    return result_ind, result_ori, result_inv

def sparse_matrix_multiply_sp(A, B, E, r_size, tau_1, is_training=False, dropout=None, is_max=False,
                              wot_i=False, weight=None):
    scatter = scatter_sum
    if is_max: scatter = scatter_max

    row_indices_ori, col_indices_ori, r_indices_ori, mask_values_ori, w_all = B
    indices_all, results_all = [], []
    indices_all_inv, results_all_inv = [], []
    indices_all_ind, results_all_ind = [], []
    batch_size = A.shape[0]
    L = w_all.shape[1]
    for i in range(batch_size):
        non_zero_ori = A[i]
        w = w_all[i].t()
        non_zero = block_is_in(row_indices_ori, non_zero_ori)
        non_zero_inv = block_is_in(col_indices_ori, non_zero_ori)

        col_indices = torch.index_select(col_indices_ori, index=non_zero, dim=0)
        mask_values = torch.index_select(mask_values_ori, index=non_zero, dim=0)
        r_indices = torch.index_select(r_indices_ori, index=non_zero, dim=0)
        if weight is not None:
            C_ori = torch.index_select(weight, index=non_zero.to(weight.device), dim=0).to(A.device)
            C_ori = score_function(C_ori)

        col_indices_inv = torch.index_select(row_indices_ori, index=non_zero_inv, dim=0)
        mask_values_inv = torch.index_select(mask_values_ori, index=non_zero_inv, dim=0)
        r_indices_inv = torch.index_select(r_indices_ori, index=non_zero_inv, dim=0)
        if weight is not None:
            C_inv = torch.index_select(weight, index=non_zero_inv.to(weight.device), dim=0).to(A.device)
            C_inv = score_function(C_inv)

        w = torch.softmax(w / tau_1, dim=0)

        B_values = torch.index_select(w[:r_size, :], index=r_indices, dim=0)
        result_values = B_values.t() * mask_values.unsqueeze(0)
        if weight is not None: result_values = result_values * C_ori.t()
        # if is_training: result_values = self.dropout(result_values)
        col_indices_uni = torch.unique(col_indices)
        sorted_indices = torch.searchsorted(col_indices_uni, col_indices)
        result = scatter(result_values, sorted_indices.long(), dim=1, dim_size=col_indices_uni.shape[0])
        # k = min(col_indices_uni.shape[0], topk)
        # _, col_indices_uni_topk = torch.topk(result.sum(0), k, dim=-1)
        index = torch.ones_like(col_indices_uni) * i
        index = torch.stack([index, col_indices_uni], dim=0)
        indices_all.append(index)
        results_all.append(result)

        # A_values = torch.index_select(A, dim=1, index=row_indices)
        B_values_inv = torch.index_select(w[r_size: 2 * r_size, :], index=r_indices_inv, dim=0)
        result_values_inv = B_values_inv.t() * mask_values_inv.unsqueeze(0)
        if weight is not None: result_values_inv = result_values_inv * C_inv.t()
        # if is_training: result_values_inv = self.dropout(result_values_inv)
        col_indices_uni_inv = torch.unique(col_indices_inv)
        sorted_indices_inv = torch.searchsorted(col_indices_uni_inv, col_indices_inv)
        result_inv = scatter(result_values_inv, sorted_indices_inv.long(), dim=1, dim_size=col_indices_uni_inv.shape[0])
        # k = min(col_indices_uni_inv.shape[0], topk)
        # _, col_indices_uni_topk_inv = torch.topk(result_inv.sum(0), k, dim=-1)
        index = torch.ones_like(col_indices_uni_inv) * i
        index = torch.stack([index, col_indices_uni_inv], dim=0)
        indices_all_inv.append(index)
        results_all_inv.append(result_inv)

        if not wot_i:
            index = torch.ones_like(A[i]) * i
            index = torch.stack([index, A[i]], dim=0)
            indices_all_ind.append(index.unsqueeze(dim=-1))
            results_all_ind.append(w[-1:, :])
    i = torch.cat(indices_all, dim=-1)
    v = torch.cat(results_all, dim=-1).t()
    output_ori = torch.sparse.FloatTensor(i.long(), v, torch.Size([batch_size, E, L]))

    i = torch.cat(indices_all_inv, dim=-1)
    v = torch.cat(results_all_inv, dim=-1).t()
    output_inv = torch.sparse.FloatTensor(i.long(), v, torch.Size([batch_size, E, L]))

    if not wot_i:
        i = torch.cat(indices_all_ind, dim=-1)
        v = torch.cat(results_all_ind, dim=0)
        output_ind = torch.sparse.FloatTensor(i.long(), v, torch.Size([batch_size, E, L]))
    else:
        output_ind = None

    return output_ind, output_ori, output_inv

def sparse_matrix_multiply_L_sp(A, B, E, r_size, tau_1, is_training=False, dropout=None, is_max=False,
                                top_k=1000, topk_pruning=100000, wot_i=False, weight=None, use_topk=False):
    scatter = scatter_sum
    if is_max: scatter = scatter_max

    row_indices_ori, col_indices_ori, r_indices_ori, mask_values_ori, w_all = B
    indices_all, results_all = [], []
    indices_all_inv, results_all_inv = [], []
    indices_all_ind, results_all_ind = [], []
    A_batch = torch.unbind(A, dim=0)
    batch_size = len(A_batch)
    L = w_all.shape[1]
    for i in range(batch_size):
        w = w_all[i].t()
        A_ = A_batch[i].coalesce()
        A_indices = A_.indices()[0]
        A_values = A_.values()
        non_zero_ori = A_indices
        if use_topk:
            k_ = min(top_k, A_values.shape[0])
            topk = torch.topk(A_values.sum(1), k=k_)[1]
            if topk.shape[0] < non_zero_ori.shape[0]:
                non_zero_ori = A_indices[topk]
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

        w = torch.softmax(w / tau_1, dim=0)

        sorted_indices = torch.searchsorted(A_indices, row_indices)
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
            # if is_training: result_values_ori = dropout(result_values_ori)
            col_indices_ori_topk = col_indices[A_values_ori_topk_indices]
            col_indices_uni = torch.unique(col_indices_ori_topk)
            sorted_indices = torch.searchsorted(col_indices_uni, col_indices_ori_topk)
            result_ori = scatter(result_values_ori, sorted_indices, dim=0, dim_size=col_indices_uni.shape[0])
        else:
            B_values_ori = torch.index_select(w[:r_size, :], index=r_indices, dim=0)
            result_values_ori = A_values_ori * B_values_ori * mask_values.unsqueeze(dim=1)
            if weight is not None:
                result_values_ori = result_values_ori * C_ori.squeeze(dim=1)
            # if is_training: result_values_ori = dropout(result_values_ori)
            col_indices_uni = torch.unique(col_indices)
            sorted_indices = torch.searchsorted(col_indices_uni, col_indices)
            result_ori = scatter(result_values_ori, sorted_indices, dim=0, dim_size=col_indices_uni.shape[0])

        # k = min(col_indices_uni.shape[0], topk)
        # _, col_indices_uni = torch.topk(result.sum(1), k, dim=0)
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
            # if is_training: result_values_inv = dropout(result_values_inv)
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
            # if is_training: result_values_inv = dropout(result_values_inv)
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

def match_constants( s, t, e2triple, constant, constant_inv, gate, gate_inv, flag):
    constant = torch.sigmoid(constant[t].t().unsqueeze(dim=0))
    gate = torch.sigmoid(gate[t]).unsqueeze(dim=0).unsqueeze(dim=-1)
    if flag:
        constant = torch.sigmoid(constant_inv[t].t().unsqueeze(dim=0))
        gate = torch.sigmoid(gate_inv[t]).unsqueeze(dim=0).unsqueeze(dim=-1)
    s_ori = s
    s *= gate
    s[:, :, e2triple[-1]] += constant * s_ori[:, :, e2triple[-1]] * (1 - gate)
    return s


def match_neibour( A, B, h, n, batch_size):
    row_indices, col_indices, r_indices = B
    results = []
    for i in range(batch_size):
        non_zero = A[i]
        indices_row = block_is_in(row_indices, non_zero)
        r_indices_row = torch.index_select(r_indices, index=indices_row, dim=0)
        indices_row_uni = torch.unique(r_indices_row, dim=0)
        # B_values = torch.index_select(h[:, :n], index=indices_row_uni, dim=1)
        B_values = torch.range(0, n - 1).to(A.device)
        B_values = torch.isin(B_values, indices_row_uni).float()

        indices_col = block_is_in(col_indices, non_zero)
        r_indices_col = torch.index_select(r_indices, index=indices_col, dim=0)
        indices_col_uni = torch.unique(r_indices_col, dim=0)
        C_values = torch.range(0, n - 1).to(A.device)
        C_values = torch.isin(C_values, indices_col_uni).float()
        D_values = torch.ones(1).to(A.device)
        #  C_values = torch.index_select(h[:, n:-1], index=indices_col_uni, dim=1)
        # constraints = (h[:, -2:-1] + (1 - h[:, -2:-1]) * activation_jit(B_values + C_values, one)).unsqueeze(dim=0)
        constraints = torch.cat([B_values, C_values, D_values], dim=0)
        constraints = torch.einsum('n,ln->l', constraints, h[i])
        # constraints = activation_jit(constraints.sum(dim=-1), one).unsqueeze(dim=0)
        results.append(constraints)
    return torch.cat(results, dim=0)

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
    # return torch.minimum(torch.relu(1 + x), torch.ones_like(x))
    return 0.5 + 0.5 * torch.sigmoid(x)
    # return torch.sigmoid(x)