#
# Copyright 2025 - IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#


import torch as t
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
import numpy as np

device = 'cuda' if t.cuda.is_available() else 'cpu'

class LSRNNBlock(nn.Module):

    def __init__(self,
                 embed_size: int,
                 hidden_size: int,
                 num_A: int = 2,
                 Lp_norm: float = 1.2,
                 **kwargs):
        
        super(LSRNNBlock, self).__init__()

        self.hidden_size = hidden_size
        self.Lp_norm = Lp_norm
        self.num_A = num_A

        self.A_selector = nn.Linear(embed_size, num_A)
        self.B = nn.Linear(embed_size, hidden_size, bias = False)

        # Xavier initialization
        initializer = t.rand(hidden_size, hidden_size, num_A)*(2*np.sqrt(6))/(np.sqrt(2*hidden_size)) - np.sqrt(6)/(np.sqrt(2*hidden_size))

        self.A_dict = nn.Parameter(initializer)        

    def forward(self, x: t.Tensor) -> t.Tensor:
        
        B, L, D = x.shape
        
        output = t.zeros(B, L, self.hidden_size).to(device)

        # B x N
        hidden_state = t.zeros(B, self.hidden_size).to(device)

        # B x L x K
        selections = F.softmax(self.A_selector(x), dim=-1)

        # B x L x N
        inputs_ = self.B(x)


        for i in range(L):
            A_weights = selections[:,i,:]
            A = einsum(self.A_dict, A_weights, 'n1 n2 k, b k -> b n1 n2')
            # Lp normalization of A
            transition_matrix = A / t.norm(A, p=self.Lp_norm, dim=1).unsqueeze(1)
            hidden_state = einsum(transition_matrix, hidden_state, 'b n1 n2, b n2 -> b n1') + inputs_[:,i,:]
            output[:,i,:] = hidden_state

        return output


class LSRNN(nn.Module):
    def __init__(self,
                 output_size: int,
                 input_size: int,
                 state_size: int,
                 num_transition_matrices: int = 2,
                 return_all_outputs: bool = False,
                 mlp_multiplier: int = 1,
                 debug: bool = False,
                 Lp_norm: float = 1.2,
                 **kwargs):
        
        super(LSRNN, self).__init__()

        self.return_all_outputs = return_all_outputs

        self.block = LSRNNBlock(embed_size=input_size,
                                hidden_size=state_size,
                                num_A=num_transition_matrices,
                                mlp_multiplier=mlp_multiplier,
                                Lp_norm=Lp_norm) 
        
        self.norm = nn.LayerNorm(state_size)

        self.readout = nn.Linear(state_size, output_size)

        self.debug = debug

    def forward(self, x: t.Tensor) -> t.Tensor:

        rnn_out = self.block(x)

        rnn_out_norm = self.norm(rnn_out)
        
        output = self.readout(rnn_out_norm)

        if not self.return_all_outputs:
            output = output[:, -1, :]
        
        if self.debug:
            return rnn_out, rnn_out_norm, output
        else:
            return output