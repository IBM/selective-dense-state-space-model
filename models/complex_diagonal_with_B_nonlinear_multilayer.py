#
# Copyright 2025 - IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#


import torch as t
import torch.nn as nn
import numpy as np
import math

device = 'cuda' if t.cuda.is_available() else 'cpu'

class LSRNNBlock(nn.Module):

    def __init__(self,
                 embed_size: int,
                 hidden_size: int,
                 scale_eigv: float = 1,
                 **kwargs):
        
        super(LSRNNBlock, self).__init__()

        self.hidden_size = hidden_size
        self.embed_size = embed_size

        self.scale_eigv = scale_eigv

        self.A_generator = nn.Sequential(
            nn.Linear(embed_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, 2 * hidden_size)
        )

        # Initial state is a random complex unit phasor
        rand_init_phases = t.rand(hidden_size)*2*math.pi
        init_state = t.exp(t.complex(real=t.zeros_like(rand_init_phases), imag=rand_init_phases))
        self.register_buffer('init_state', init_state, persistent=True)

        # Xavier initialization with halved variance
        self.B_re = nn.Parameter(t.rand(hidden_size, embed_size)*(2*np.sqrt(3))/(np.sqrt(hidden_size+embed_size)) - np.sqrt(3)/(np.sqrt(hidden_size+embed_size)))
        self.B_im = nn.Parameter(t.rand(hidden_size, embed_size)*(2*np.sqrt(3))/(np.sqrt(hidden_size+embed_size)) - np.sqrt(3)/(np.sqrt(hidden_size+embed_size)))
        

    def forward(self, x: t.Tensor) -> t.Tensor:

        B, L, D = x.shape

        output = t.complex(real = t.zeros(B, L, self.hidden_size), imag= t.zeros(B, L, self.hidden_size)).to(device)

        hidden_state = self.init_state.view(1, self.hidden_size).repeat(B, 1)

        # B x L x 2H
        transition_matrices_real_im = self.A_generator(x)

        # B x L x H
        transition_matrices = t.complex(real = transition_matrices_real_im[:, :, :self.hidden_size], imag = transition_matrices_real_im[:, :, self.hidden_size:])

        # H x D
        B_mat = t.complex(real = self.B_re, imag = self.B_im)

        # B x L x H
        inputs = t.matmul(B_mat.view(1, 1, self.hidden_size, self.embed_size).expand(B, L, -1, -1), t.complex(real=x, imag=t.zeros_like(x)).view(B,L,D, 1)).squeeze(-1)

        # Normalize then scale the transition matrices
        transition_matrices = (transition_matrices / t.abs(transition_matrices)) * self.scale_eigv

        for i in range(L):

            # Update the hidden state
            hidden_state = transition_matrices[:,i,:] * hidden_state + inputs[:,i,:]
            
            # Put it into the output tensor
            output[:,i,:] = hidden_state

        return output

class LSRNN(nn.Module):
    def __init__(self,
                 output_size: int,
                 input_size: int,
                 state_size: int,
                 num_transition_matrices: int = 2,
                 return_all_outputs: bool = False,
                 mlp_multiplier: int = 2,
                 debug: bool = False,
                 num_layers: int = 4,
                 ignore_empty_token: bool = False,
                 eps=1e-7,
                 scale_eigv: float = 1,
                 embed_size: int = 32,
                 **kwargs):
        
        
        super(LSRNN, self).__init__()

        self.return_all_outputs = return_all_outputs
        self.output_size = output_size
        self.state_size = state_size
        self.ignore_empty_token = ignore_empty_token
        self.eps = eps
        self.num_layers = num_layers

        self.embedding = nn.Linear(input_size, embed_size, bias=False)

        blocks = [LSRNNBlock(embed_size=embed_size,
                                hidden_size=state_size,
                                num_A=num_transition_matrices,
                                mlp_multiplier=mlp_multiplier,
                                scale_eigv=scale_eigv)]

        for _ in range(num_layers - 1):

            blocks.append(LSRNNBlock(embed_size=state_size,
                                hidden_size=state_size,
                                num_A=num_transition_matrices,
                                mlp_multiplier=mlp_multiplier,
                                scale_eigv=scale_eigv))

        self.blocks = nn.ModuleList(blocks)

        
        mlps = []

        # Intermediate MLPs
        for _ in range(num_layers - 1):
            mlps.append(nn.Sequential(
            nn.Linear(2*state_size, 2*state_size),
            nn.ReLU(),
            nn.Linear(2*state_size, state_size)))
        
        # Readout
        mlps.append(nn.Sequential(
            nn.Linear(2*state_size, 2*state_size),
            nn.ReLU(),
            nn.Linear(2*state_size, output_size)))

        self.mlps = nn.ModuleList(mlps)

        self.debug = debug

    def forward(self, x: t.Tensor) -> t.Tensor:

        x = self.embedding(x).to(device)

        for i in range(self.num_layers):
            if i == 0:
                rnn_out_complex = self.blocks[i](x)
            else:
                rnn_out_complex = self.blocks[i](rnn_out_complex)

            rnn_out_complex =  t.cat((rnn_out_complex.real, rnn_out_complex.imag), dim=-1)

            rnn_out_complex = self.mlps[i](rnn_out_complex)

        # B x L x D
        output = rnn_out_complex[:,-1,:]
        
        return output