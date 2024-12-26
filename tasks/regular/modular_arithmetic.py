#
# Copyright 2025 - IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#


import functools
from typing import Optional, Sequence

import jax
import jax.numpy as jnp

import torch as t
import torch.nn.functional as F
import numpy as np

from State_Tracking_With_NNs.tasks import task

# Public as this may be used to encode/decode strings of numbers/symbols.
OP_BY_CHARACTER = {'+': 0, '-': 1, '*': 2, '_': 3}

def _replace_subtractions(expression: jnp.ndarray, modulus: int) -> jnp.ndarray:
  """Replaces subtractions in an expression by additions with the inverse.

  e.g. the expression [1, -, 3] results in [1, +, -3].

  Args:
    expression: Encoded expression (a 1D array of integers) in which to replace
      subtractions.
    modulus: The modulus to use for the modular arithmetic.

  Returns:
    The expression with all subtractions replaced by additions with the inverse.
  """
  if expression.size < 2:
    return expression

  mask = (expression == modulus + OP_BY_CHARACTER['-'])
  subtract_replaced = jnp.where(mask, modulus + OP_BY_CHARACTER['+'],
                                expression)
  return subtract_replaced.at[2:].multiply(1 - 2 * mask[1:-1])


def _perform_multiplications(expression: jnp.ndarray,
                             modulus: int) -> jnp.ndarray:
  """Performs all multiplications in an expression containing only + and *.

  This is done at fixed length and the result is zero-padded to achieve this.
  Since the result of performing multiplications is an expression containing
  only + operators, the operators are dropped from the output. For example, the
  expression [1, +, 3, *, 4] results in [1, 12, 0].

  Args:
    expression: Encoded expression in which to perform multiplications.
    modulus: The modulus to use for the modular arithmetic.

  Returns:
    An array with the results of the multiplications (potentially zero-padded).
  """
  term_ids = jnp.cumsum(expression == modulus + OP_BY_CHARACTER['+'])[::2]
  # Segment_prod can only be jit-compiled with a fixed number of segments.
  # Therefore, we have to set to the maximum number of terms possible and
  # mask out superfluous segment results with zeros afterwards.
  maximum_term_number = expression.shape[0] // 2 + 1
  products = jax.ops.segment_prod(
      expression[::2],
      term_ids,
      num_segments=maximum_term_number,
      indices_are_sorted=True)
  valid_segment_mask = jnp.arange(maximum_term_number) <= term_ids[-1]
  return products * valid_segment_mask


def _replace_blanks(expression: jnp.ndarray, modulus: int) -> jnp.ndarray:
  """Replaces blank symbols in expression with either `+` or `0`.

  Depending on whether the blank symbol is at the position of an operator or a
  residual, the blank symbol is replaced with a `+` operator or a `0`.

  Args:
    expression: Encoded expression in which to replace blank symbols.
    modulus: The modulus to use for the modular arithmetic.

  Returns:
    An array with blank symbols replaced by either `+` or `0`.
  """
  mask = (expression == OP_BY_CHARACTER['_'] + modulus)
  operator_mask = mask.at[::2].set(False)
  residual_mask = mask.at[1::2].set(False)

  blanks_replaced = jnp.where(operator_mask, OP_BY_CHARACTER['+'] + modulus,
                              expression)
  blanks_replaced = jnp.where(residual_mask, 0, blanks_replaced)
  return blanks_replaced


def _evaluate_expression(expression: jnp.ndarray, modulus: int) -> jnp.ndarray:
  """Returns the result of evaluating a modular arithmetic expression."""
  expression = _replace_blanks(expression, modulus)
  expression = _replace_subtractions(expression, modulus)
  additive_terms = _perform_multiplications(expression, modulus)
  return jnp.sum(additive_terms) % modulus


class ModularArithmetic(task.GeneralizationTask):
  """A task with the goal of reducing a simple arithmetic expression.

  The input is a string, composed of numbers (in {0, ..., modulus-1}), and
  operators (in {+, -, *}). The output is the reduced value of this expression,
  which is also in {0, ..., modulus-1}.

  Examples (modulo 5):
    1 + 2 * 3 = 2
    1 - 1 - 1 = 4
    0 * 1 + 4 * 3 - 2 = 0

  Note that the input strings are always of odd length.
  """

  def __init__(self,
               modulus: int = 5,
               *args,
               operators: Optional[Sequence[str]] = None,
               **kwargs):
    """Initializes the modular arithmetic task.

    Args:
      modulus: The modulus used for the computation.
      *args: Args for the base task class.
      operators: Operators to be used in the sequences. By default it's None,
        meaning all operators available are used.
      **kwargs: Kwargs for the base task class.
    """
    super().__init__(*args, **kwargs)

    self._modulus = modulus
    if operators is None:
      operators = ('+', '*', '-')

    self._operators = (0, 2, 1)

  #  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(
      self,
      batch_size: int,
      length: int,
  ) -> task.Batch:
    """Returns a batch of modular arithmetic expressions and their labels.

    Args:
      rng: The jax random number generator.
      batch_size: The size of the batch returned.
      length: The length of the sequence. As this length must be odd for the
        modular arithmetic dataset, if it's not, we force it to be by
        subtracting one to the length passed.
    """
    # Subtracting one to the length if it's not odd already.
    if length % 2 != 1:
      length -= 1

    batch = jnp.empty((batch_size, length), dtype=int)
    remainders = np.random.randint(0, self._modulus, (batch_size, length // 2 + 1))
    remainders = jnp.array(remainders)
    ops = self._modulus + np.array(list(self._operators))

    operations = np.random.choice(ops, (batch_size, length // 2))
    operations = jnp.array(operations)

    batch = batch.at[:, ::2].set(remainders)
    expressions = batch.at[:, 1::2].set(operations)

    evaluate = functools.partial(_evaluate_expression, modulus=self._modulus)
    labels = t.from_numpy(np.array(jax.vmap(evaluate)(expressions))).long()
    expressions = t.from_numpy(np.array(expressions)).long()

    labels = F.one_hot(labels, self._modulus)
    one_hot_expressions = F.one_hot(expressions,
                                      self._modulus + len(OP_BY_CHARACTER))
    
    
    return {'input': one_hot_expressions.float(), 'output': labels}

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return self._modulus + len(OP_BY_CHARACTER)

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return self._modulus