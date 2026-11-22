import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class LinearAttn(nn.Module):
    def __init__(self, output_dim, key_dim):
        super(LinearAttn, self).__init__()

        # variables
        self.W = nn.Linear(output_dim, key_dim, bias=True)
        self.q = nn.Linear(key_dim, 1, bias=False)
        self.tanh = nn.Tanh()

        # initialize
        self.W.apply(init_weights)
        self.q.apply(init_weights)

    def forward(self, inputs):
        # inputs: [batch, len_seq, (head_num * head_dim)]
        # attn: [batch, len_seq, 1]
        attn = self.q(self.tanh(self.W(inputs)))
        attn = F.softmax(attn, dim=-2)

        # output: [batch, (head_num * head_dim)]
        output = torch.mul(inputs, attn).sum(-2)
        return output


class GlobalAttn(nn.Module):
    def __init__(self, his_dim, global_dim):
        super(GlobalAttn, self).__init__()

        # variables
        # self.W = nn.Linear(his_dim, global_dim, bias=False)
        self.W1 = nn.Linear(his_dim + global_dim, 100)
        self.W2 = nn.Linear(100, 50)
        self.W3 = nn.Linear(50, 1)
        self.acti = nn.ReLU()

        # initialize
        # self.W.apply(init_weights)

    def forward(self, his, global_pref):
        # his: [batch, len_seq, his_dim]
        # global_pref: [batch, global_dim]
        # attn: [batch, len_seq, 1]
        # attn = torch.matmul(self.W(his), global_pref.unsqueeze(2))
        attn = torch.cat((his, global_pref.unsqueeze(1).repeat(1, his.shape[1], 1)), dim=-1)
        attn = self.W3(self.acti(self.W2(self.acti(self.W1(attn)))))
        attn = F.softmax(attn, dim=-2)

        output = torch.mul(his, attn).sum(-2)
        return output


class SelfAttn(nn.Module):
    def __init__(self, head_num, head_dim, input_dim, mask_right=False):
        """Initialization for variables in SelfAttention.
        WQ is used for linear transformation of query.
        WK is used for linear transformation of key.
        WV is used for linear transformation of value.
        """
        super(SelfAttn, self).__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.multihead_dim = head_num * head_dim
        self.input_dim = input_dim
        self.mask_right = mask_right

        # variables
        self.WQ = nn.Linear(self.input_dim, self.multihead_dim, bias=False)
        self.WK = nn.Linear(self.input_dim, self.multihead_dim, bias=False)
        self.WV = nn.Linear(self.input_dim, self.multihead_dim, bias=False)

        # initialize
        self.WQ.apply(init_weights)
        self.WK.apply(init_weights)
        self.WV.apply(init_weights)

    def forward(self, QKV, mask=None):
        Q_seq, K_seq, V_seq = QKV
        num_mode = len(Q_seq.shape)
        if num_mode == 4:   # user encoder
            # seq: [batch, len_seq1, len_seq2, emb_dim]
            # QKV: [batch, len_seq1, len_seq2, head_num, head_dim]
            Q = self.WQ(Q_seq).view(-1, Q_seq.shape[1], Q_seq.shape[2], self.head_num, self.head_dim)
            K = self.WK(K_seq).view(-1, K_seq.shape[1], K_seq.shape[2], self.head_num, self.head_dim)
            V = self.WV(V_seq).view(-1, V_seq.shape[1], V_seq.shape[2], self.head_num, self.head_dim)

            # QKV: [batch, len_seq1, head_num, len_seq2, head_dim]
            Q, K, V = Q.transpose(2, 3), K.transpose(2, 3), V.transpose(2, 3)

            # attn: [batch, len_seq1, head_num, len_seq2_Q, len_seq2_K]
            norm = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
            attn = torch.matmul(Q, K.transpose(3, 4)) / norm
            if mask is not None:
                attn = attn.masked_fill(mask == 0, -1e9)

            # output: [batch, len_seq1, head_num, len_seq2, head_dim]
            attn = F.softmax(attn, dim=-1)
            output = torch.matmul(attn, V)

            # output: [batch, len_seq1, len_seq2, (head_num * head_dim)]
            output = output.transpose(2, 3)
            output = output.reshape(-1, output.shape[1], output.shape[2], self.multihead_dim)
            return output

        elif num_mode == 3: # news encoder
            # seq: [batch, len_seq, emb_dim]
            # QKV: [batch, len_seq, head_num, head_dim]
            Q = self.WQ(Q_seq).view(-1, Q_seq.shape[1], self.head_num, self.head_dim)
            K = self.WK(K_seq).view(-1, K_seq.shape[1], self.head_num, self.head_dim)
            V = self.WV(V_seq).view(-1, V_seq.shape[1], self.head_num, self.head_dim)

            # QKV: [batch, head_num, len_seq, head_dim]
            Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

            # attn: [batch, head_num, len_seq_Q, len_seq_K]
            norm = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
            attn = torch.matmul(Q, K.transpose(2, 3)) / norm
            if mask is not None:
                attn = attn.masked_fill(mask == 0, -1e9)

            # output: [batch, head_num, len_seq, head_dim]
            attn = F.softmax(attn, dim=-1)
            output = torch.matmul(attn, V)

            # output: [batch, len_seq, (head_num * head_dim)]
            output = output.transpose(1, 2)
            output = output.reshape(-1, output.shape[1], self.multihead_dim)
            return output
