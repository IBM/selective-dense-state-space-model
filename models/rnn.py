#
# Copyright 2025 - IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#


import torch as t
import torch.nn as nn

class ElmanRNN(nn.Module):
    def __init__(self,
                 output_size: int,
                 input_size: int,
                 state_size: int,
                 num_layers: int = 1,
                 return_all_outputs: bool = False,
                 bidirectional: bool = False,
                 nonlinearity: str = 'relu',
                 **kwargs
                 ):
        
        super(ElmanRNN, self).__init__()
        
        self.return_all_outputs = return_all_outputs

        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=state_size,
                          num_layers=num_layers,
                          nonlinearity=nonlinearity,
                          bias=True,
                          batch_first=True,
                          dropout=0,
                          bidirectional=bidirectional)

        self.output_projection = nn.Linear(state_size, output_size, bias=False)

    def forward(self, 
                x: t.Tensor
                ) -> t.Tensor:

      rnn_out = self.rnn(x)

      # PyTorch RNN returns (output, h_n)
      if not self.return_all_outputs:
        output = rnn_out[0][:,-1]
      else:
         output = rnn_out[0]

      return self.output_projection(output)
