#
# Copyright 2025 - IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Compute whether the number of 1s in a string is even."""

import torch as t
import torch.nn.functional as F
from State_Tracking_With_NNs.tasks import task

class ParityCheck(task.GeneralizationTask):

  def sample_batch(self, batch_size: int, length: int) -> task.Batch:
    """Returns a batch of strings and the expected class."""
    strings = t.randint(low=0, high=2, size=(batch_size, length))
    n_b = t.sum(strings, axis=1) % 2
    n_b = F.one_hot(n_b, num_classes = 2)
    one_hot_strings = F.one_hot(strings, num_classes = 2)
    return {"input": one_hot_strings.float(), "output": n_b}

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 2

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 2
