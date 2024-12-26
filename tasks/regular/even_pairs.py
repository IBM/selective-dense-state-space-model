#
# Copyright 2025 - IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Compute whether the number of 01's and 10's is even."""


import torch as t
import torch.nn.functional as F
from State_Tracking_With_NNs.tasks import task


class EvenPairs(task.GeneralizationTask):

  # @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(self, batch_size: int,
                   length: int) -> task.Batch:
    """Returns a batch of strings and the expected class."""
    strings = t.randint(
        low=0,
        high=2,
        size=(batch_size, length),
    )
    one_hot_strings = F.one_hot(strings, num_classes=2)
    unequal_pairs = t.logical_xor(strings[:, :-1], strings[:, 1:])
    odd_unequal_pairs = t.sum(unequal_pairs, axis=-1) % 2
    return {
        'input': one_hot_strings.float(),
        'output': F.one_hot(odd_unequal_pairs, num_classes=self.output_size),
    }

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 2

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 2
