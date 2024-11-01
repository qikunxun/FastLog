import torch
from torch import nn
from utils import sparse_matrix_multiply_sp, sparse_matrix_multiply_L_sp, log_loss_common_sp, norm_sp, sum_sp

device = torch.device('cuda')
class Model(nn.Module):
    def __init__(self, n, T, L, E, N, emb_size, tau_1=10, tau_2=0.2, mask_w=False, use_gpu=False,
                 dropout=0.1, top_k=1000, top_k_mask=100000, use_soft=False, use_topk=False):
        super(Model, self).__init__()
        self.T = T
        self.L = L
        self.E = E
        self.n = n - 1
        self.r_size = (n - 1) // 2
        self.N = N
        self.use_soft = use_soft
        self.use_topk = use_topk
        self.emb_size = emb_size
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.mask_w = mask_w
        self.top_k = top_k
        self.top_k_mask = top_k_mask
        self.emb = nn.Embedding(self.n + 1, self.emb_size)
        self.lstm = torch.nn.LSTM(
                self.emb_size, self.emb_size,
                1
            )
        self.linear = nn.Linear(self.emb_size, self.n)
        if self.use_soft:
            self.weight = nn.Parameter(torch.Tensor(self.N, 1))
            nn.init.zeros_(self.weight)
        else:
            self.weight = None

        self.use_gpu = use_gpu
        self.dropout = nn.Dropout(dropout)
        self.one = torch.autograd.Variable(torch.Tensor([1])).detach()
        if use_gpu:
            self.one = self.one.cuda()

    def forward(self, input_x, input_r, e2triple, triple2e, r2triple, is_training=False, input_y=None):
        batch_size = input_x.shape[0]
        N = triple2e[1].shape[0]
        E = self.E
        i = torch.stack([torch.arange(0, batch_size).to(input_x.device), input_x], dim=0)
        v = torch.ones_like(input_x).unsqueeze(dim=-1)
        x_ori_i = torch.sparse.FloatTensor(i, v, torch.Size([batch_size, E, 1]))
        input_emb_ori = self.emb(input_r)
        input_emb = torch.stack([input_emb_ori] * (self.T + 1), dim=1)
        input_emb[:, -1, :] = self.emb(torch.ones_like(input_r) * self.n)

        rnn_outputs, _ = self.lstm(input_emb.transpose(1, 0))
        rnn_outputs = rnn_outputs.transpose(1, 0)
        w_all = self.linear(rnn_outputs[:, :-1, :])
        states = [x_ori_i]
        for t in range(self.T + 1):
            a = rnn_outputs[:, t]
            b = rnn_outputs[:, :t + 1]
            x = torch.einsum("bd, btd -> bt", a, b)
            attention = torch.softmax(x, dim=-1)
            input = []
            for j, state in enumerate(states):
                input.append(self.match_att(state, attention[:, j]))
            input = torch.stack(input, dim=0)
            input = torch.sparse.sum(input, dim=0)
            if t < self.T:
                w_probs = w_all[:, t, :].unsqueeze(dim=1)
                if t == 0:
                    _, s_h, s_t = sparse_matrix_multiply_sp(input_x,
                                        (e2triple[0], triple2e[1], r2triple[0], e2triple[2], w_probs),
                                        E, self.r_size, self.tau_1, is_training, wot_i=True,
                                        weight=self.weight)
                    s = s_h + s_t
                    # if is_training: s = self.dropout(s)
                if t >= 1:
                    x = input  # [b, L, E]
                    _, s_h, s_t = sparse_matrix_multiply_L_sp(x,
                                        (e2triple[0], triple2e[1], r2triple[0], e2triple[2], w_probs),
                                        E, self.r_size, self.tau_1, is_training, self.dropout,
                                        top_k=self.top_k, topk_pruning=self.top_k_mask,
                                        wot_i=True, weight=self.weight, use_topk=self.use_topk)
                    s = s_h + s_t

                    # if is_training: s = self.dropout(s)
                s = norm_sp(s)
            else:
                s = input
            states.append(s)
        state = states[-1]
        s = sum_sp(state)
        return s

    def log_loss(self, p_score, label):
        return log_loss_common_sp(p_score, label, self.E, self.tau_2)

    def match_att(self, state, att):
        A = state.coalesce()
        indices = A.indices()
        values = A.values()
        att_exp = torch.index_select(att, dim=0, index=indices[0])
        values = values * att_exp.unsqueeze(dim=-1)
        return torch.sparse.FloatTensor(indices, values, A.shape)