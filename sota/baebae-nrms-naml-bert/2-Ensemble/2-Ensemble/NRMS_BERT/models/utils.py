######################################################################################################
# Mind 2020
# Authors: Hyunsik Jeon(jeon185@snu.ac.kr), SeungCheol Park(ant6si@snu.ac.kr),
#          Yuna Bae(yunabae482@gmail.com), U kang(ukang@snu.ac.kr)
# File: src/NRMS_BERT/models/utils.py
# - Utility functions for the model construction.
#
# Version: 1.0
#######################################################################################################
import sys
sys.path.append("../")
sys.path.append("./")

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    """
    Initialize the weights for the linear layer
    :param m: module
    :return:
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


class LinearAttn(nn.Module):
    """
    Linear attention
    """
    def __init__(self, output_dim, key_dim):
        """
        Initialization
        :param output_dim: dimension of the output
        :param key_dim: dimension of the key
        """
        super(LinearAttn, self).__init__()

        # variables
        self.W = nn.Linear(output_dim, key_dim, bias=True)
        self.q = nn.Linear(key_dim, 1, bias=False)
        self.tanh = nn.Tanh()

        # initialize
        self.W.apply(init_weights)
        self.q.apply(init_weights)

    def forward(self, inputs):
        """
        Get output
        :param inputs:
        :return: output
        """
        # inputs: [batch, len_seq, (head_num * head_dim)]
        # attn: [batch, len_seq, 1]
        attn = self.q(self.tanh(self.W(inputs)))
        attn = F.softmax(attn, dim=-2)

        # output: [batch, (head_num * head_dim)]
        output = torch.mul(inputs, attn).sum(-2)
        return output


class GlobalAttn(nn.Module):
    """
    Class for global attention
    """
    def __init__(self, his_dim, global_dim):
        """
        Initialize the global attention class
        :param his_dim: dimension of the history
        :param global_dim: dimension of the global
        """
        super(GlobalAttn, self).__init__()

        # variables
        self.W = nn.Linear(his_dim, global_dim, bias=False)

        # initialize
        self.W.apply(init_weights)

    def forward(self, his, global_pref):
        # his: [batch, len_seq, his_dim]
        # global_pref: [batch, global_dim]
        # attn: [batch, len_seq, 1]
        attn = torch.matmul(self.W(his), global_pref.unsqueeze(2))
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


import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(torch.nn.Module):
    """
    A general additive attention module.
    Originally for NAML.
    """
    def __init__(self,
                 query_vector_dim,
                 candidate_vector_dim,
                 writer=None,
                 tag=None,
                 names=None):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.attention_query_vector = nn.Parameter(
            torch.empty(query_vector_dim).uniform_(-0.1, 0.1))
        # For tensorboard
        self.writer = writer
        self.tag = tag
        self.names = names
        self.local_step = 1

    def forward(self, candidate_vector):
        """
        Args:
            candidate_vector: batch_size, candidate_size, candidate_vector_dim
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        # batch_size, candidate_size, query_vector_dim
        temp = torch.tanh(self.linear(candidate_vector))
        # batch_size, candidate_size
        candidate_weights = F.softmax(torch.matmul(
            temp, self.attention_query_vector),
                                      dim=1)
        if self.writer is not None:
            assert candidate_weights.size(1) == len(self.names)
            if self.local_step % 10 == 0:
                self.writer.add_scalars(
                    self.tag, {
                        x: y
                        for x, y in zip(self.names,
                                        candidate_weights.mean(dim=0))
                    }, self.local_step)
            self.local_step += 1
        # batch_size, candidate_vector_dim
        target = torch.bmm(candidate_weights.unsqueeze(dim=1),
                           candidate_vector).squeeze(dim=1)
        return target