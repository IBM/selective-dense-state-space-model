#
# Copyright 2025 - IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#


import torch as t
import numpy as np
import torch.nn.functional as F
import itertools
import math
from sympy.combinatorics.permutations import Permutation

from selective_dense_state_space_model.tasks import task

class A5Navigation(task.GeneralizationTask):
  """
  Traversal of the A5 group's diagram, taken from Bingbin Liu's code.
  """

  def __init__(self, **kwargs):
    
    # A5 group
    self.n = 5

    assert self.n == 5, "Let's please use the A5 automaton"

    # Id / flip first two / cycle
    self.n_actions = 2

    self.label_type = 'state'

    """
    Get states
    """
    self.state_label_map = {}

    state_encode_temp = lambda state: ''.join([str(int(each)) for each in state])

    cnt = 0
    for si, state in enumerate(itertools.permutations(range(self.n))):
        if not Permutation(state).is_even:
                continue
        enc = state_encode_temp(state)
        self.state_label_map[enc] = cnt
        cnt += 1

    # The first action, cycle
    action_cycle = t.roll(t.eye(self.n), shifts=-1, dims=0).view(1, self.n, self.n)

    # The second action, swapping axes
    identity = t.eye(self.n)
    index = t.Tensor([1, 0, 3, 2, 4]).long()
    action_flip = identity[index].view(1, self.n, self.n)

    self.actions = t.cat((action_cycle, action_flip), dim=0)

  def state_encode(self, state):
    # State of shape B x D
    B, D = state.shape

    state_encoding = []

    for b in range(B):
      curr_state = state[b, :]
      state_str = ''

      for el in curr_state:        
        state_str += str(int(el))

      state_encoding.append(state_str)
             
    return state_encoding
            
  def get_state_label(self, state):
    enc = self.state_encode(state)

    labels = []
    
    for encoding in enc:
       labels.append(self.state_label_map[encoding])

    return t.Tensor(labels)

  def f(self, x):
    B, L, D = x.shape
    curr_state = t.arange(self.n).view(1, self.n).repeat(B, 1).float()
    labels = []

    actions = t.einsum('bld, dmn -> blmn', x, self.actions)

    for action_idx in range(L):

      # B x M x M
      curr_action = actions[:,action_idx,:,:]

      # B x M
      curr_state = t.einsum('bn, bmn -> bm', curr_state, curr_action)

    return self.get_state_label(curr_state)
    #return np.array(labels)
  
  def sample_batch(self, batch_size: int,
                   length: int) -> task.Batch:
    """Returns a batch of strings and the expected class."""

    actions = t.randint(
        size=(batch_size, length), low=0, high=self.n_actions)
    
    one_hot_strings = F.one_hot(actions, num_classes=self.input_size).float()
    
    states = self.f(x=one_hot_strings)

    final_state = states


    final_states = F.one_hot(final_state.long(), num_classes=self.output_size)
  
    return {"input": one_hot_strings, "output": final_states}
  
  @property
  def input_size(self):
    return self.n_actions
  
  @property
  def output_size(self):
    return math.factorial(self.n) // 2