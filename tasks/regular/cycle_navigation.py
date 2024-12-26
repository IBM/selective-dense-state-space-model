#
# Copyright 2025 - IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Compute the final state after randomly walking on a circle."""

import torch as t
import torch.nn.functional as F

from State_Tracking_With_NNs.tasks import task

class CycleNavigation(task.GeneralizationTask):

  @property
  def _cycle_length(self) -> int:
    return 5

  # @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(self, batch_size: int,
                   length: int) -> task.Batch:
    """Returns a batch of strings and the expected class."""
    actions = t.randint(
        size=(batch_size, length), low=0, high=3)
    final_states = t.sum(actions - 1, axis=1) % self._cycle_length
    final_states = F.one_hot(final_states, num_classes=self.output_size)
    one_hot_strings = F.one_hot(actions, num_classes=self.input_size)
    return {"input": one_hot_strings.float(), "output": final_states}

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 3

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return self._cycle_length
