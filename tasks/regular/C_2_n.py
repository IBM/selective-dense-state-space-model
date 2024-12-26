#
# Copyright 2025 - IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Compute the final state after randomly walking on a circle."""


import torch as t
import torch.nn.functional as F

from State_Tracking_With_NNs.tasks import task


class C_2_n(task.GeneralizationTask):
  """
  Traversal of the 2xn cyclic group's diagram, adapted from Bingbin Liu's code.
  """

  def __init__(self, n=4):
    
    # Dn group
    self.n = n
    # Move / toggle
    self.n_actions = 2

    self.label_type = 'state'

  def get_state_label(self, state):
        """
        toggle in {0,1}
        position in [k]
        """
        toggle, position = state
        label = self.n*toggle + position
        return label

  def f(self, x):
    # Parallel solution

    # Get toggles: a parity task on the toggle bit
    # toggles = (x == 0).astype(np.int64)
    toggles = (x == 0)
    toggle_status = t.cumsum(toggles, dim=-1) % 2

    # Get positions: a directed modular counter
    directions = (1)**toggle_status
    # directed_drives = (x != 0).astype(np.int64) * directions
    directed_drives = (x != 0) * directions
    positions = t.cumsum(directed_drives, dim=-1) % self.n

    state_label = toggle_status*self.n + positions

    return state_label



  # @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(self, batch_size: int,
                   length: int) -> task.Batch:
    """Returns a batch of strings and the expected class."""


    actions = t.randint(
        size=(batch_size, length), low=0, high=self.n_actions)
    
    states = self.f(x=actions)

    final_state = states[:,-1]

    final_states = F.one_hot(final_state, num_classes=self.output_size)
    one_hot_strings = F.one_hot(actions, num_classes=self.input_size)

    return {"input": one_hot_strings.float(), "output": final_states}


  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    # Toggle/flip
    return self.n_actions

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 2*self.n
