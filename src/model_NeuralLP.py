import torch
from torch import nn
from utils import sparse_matrix_multiply, sparse_matrix_multiply_L_sample, log_loss

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
        x_ori_i = input_x
        x_ori_i = torch.nn.functional.one_hot(x_ori_i.long(), self.E).bool()
        input_emb_ori = self.emb(input_r)
        input_emb = torch.stack([input_emb_ori] * (self.T + 1), dim=1)
        input_emb[:, -1, :] = self.emb(torch.ones_like(input_r) * self.n)

        rnn_outputs, _ = self.lstm(input_emb.transpose(1, 0))
        rnn_outputs = rnn_outputs.transpose(1, 0)
        w_all = self.linear(rnn_outputs[:, :-1, :])
        states = [x_ori_i.to(rnn_outputs).unsqueeze(dim=1)]
        for t in range(self.T + 1):
            a = rnn_outputs[:, t]
            b = rnn_outputs[:, :t + 1]
            x = torch.einsum("bd, btd -> bt", a, b)
            attention = torch.softmax(x, dim=-1)
            memory = torch.stack(states, dim=0)
            input = torch.einsum("bt, tbln -> bln", attention, memory)

            if t < self.T:
                w_probs = w_all[:, t, :].unsqueeze(dim=1)
                if t == 0:
                    _, s_h, s_t = sparse_matrix_multiply(input.squeeze(dim=1),
                                        (e2triple[0], triple2e[1], r2triple[0], e2triple[2], w_probs),
                                        E, self.r_size, self.tau_1, is_training, weight=self.weight, wot_i=True)
                    s = s_h + s_t
                if t >= 1:
                    x = input  # [b, L, E]
                    _, s_h, s_t = sparse_matrix_multiply_L_sample(x,
                                        (e2triple[0], triple2e[1], r2triple[0], e2triple[2], w_probs),
                                        E, self.r_size, self.tau_1, is_training, self.dropout,
                                        top_k=self.top_k, topk_pruning=self.top_k_mask,
                                        weight=self.weight, use_topk=self.use_topk, wot_i=True)
                    s = s_h + s_t

                s_sum = s.sum(dim=-1, keepdims=True)
                s = s / s_sum.clamp(1e-7)
                if is_training: s = self.dropout(s)
            else:
                s = input
            states.append(s)
        state = states[-1]
        s = state.sum(dim=1)
        return s

    def log_loss(self, p_score, label):
        return log_loss(p_score, label, self.E, self.tau_2, self.one)