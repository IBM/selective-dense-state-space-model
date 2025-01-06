#
# Copyright 2025 - IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Training loop for length generalization experiments."""

import dataclasses
from typing import Any, Callable, Mapping, Optional

import torch as t
import torch.nn as nn
from torch import optim

import numpy as np
import tqdm
import os

from selective_dense_state_space_model.experiments import curriculum as curriculum_lib
from selective_dense_state_space_model.experiments import range_evaluation
from selective_dense_state_space_model.tasks import task as task_lib
from selective_dense_state_space_model.experiments import utils
from torch.utils.tensorboard import SummaryWriter

_LossMetrics = Optional[Mapping[str, t.Tensor]]
device = 'cuda' if t.cuda.is_available() else 'cpu'

@dataclasses.dataclass
class ClassicTrainingParams:
  """Parameters needed to train classical architectures."""
  
  seed: int 
  training_steps: int
  log_frequency: int

  task: task_lib.GeneralizationTask
  length_curriculum: curriculum_lib.Curriculum
  batch_size: int

  weight_decay: float
  single_output: bool

  model: nn.Module
  loss_fn: Callable[[t.Tensor, t.Tensor], tuple[float, _LossMetrics]]
  learning_rate: float
  test_model: Optional[nn.Module] = None
  max_grad_norm: float = 1.

  tboard_logdir: str = None

  range_test_total_batch_size: int = 512
  range_test_sub_batch_size: int = 64
  max_range_test_length: int = 100

  use_query_token: bool = False

  accuracy_fn: Optional[Callable[[t.Tensor, t.Tensor],
                                 t.Tensor]] = None

  # Path for storing model checkpoints
  save_path: str = None

  state_size: int = None
  embed_size: int = None
  num_transition_matrices: int = None

  train_length: int = 0

  convergence_steps: int = 10000

class TrainingWorker:
  """Training worker."""

  def __init__(self,
               training_params: ClassicTrainingParams,
               use_tqdm: bool = True,
               computation_steps_mult: int = 0,
               ):
    """Initializes the worker.

    Args:
      training_params: The training parameters.
      use_tqdm: Whether to add a progress bar to stdout.
    """
    self._training_params = training_params
    self._use_tqdm = use_tqdm
    self._computation_steps_mult = computation_steps_mult
    if training_params.tboard_logdir is not None:
      self.writer = SummaryWriter(log_dir=training_params.tboard_logdir)

  def run(
      self,
  ) -> tuple[
      list[Mapping[str, Any]], Optional[list[Mapping[str, Any]]], t.Tensor
    ]:

    """
    Trains the model with the provided config.

    Returns:
      Results (various training and validation metrics), module parameters
      and router parameters.
    """
    training_params = self._training_params

    save_path = training_params.save_path

    results = []

    model = training_params.model.to(device)

    optimizer = optim.Adam(model.parameters(), 
                            lr=training_params.learning_rate,
                            weight_decay=training_params.weight_decay,
                            )
      

    initial_step = 0
    finished = False
    checkpoint_name = None

    # Check whether the experiment is finished
    for filename in sorted(os.listdir(save_path)):
      if filename.endswith('fin'):
        finished=True

    if finished:
      for filename in sorted(os.listdir(save_path)):
        if filename.startswith('step.pt'):
          checkpoint_name = filename
    else:
      step = -1000
      for filename in sorted(os.listdir(save_path)):
        if filename.endswith('.pt') and 'step' in filename:
          # Check if it is a later checkpoint
          if int(filename.split('step_')[1].split('.pt')[0]) > step:
            checkpoint_name = filename

    # If a stored checkpoint was found
    if checkpoint_name is not None and not finished:
      checkpoint = t.load(os.path.join(save_path, checkpoint_name))
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      initial_step = checkpoint['step']

    task = training_params.task
    length_curriculum = training_params.length_curriculum
    loss_fn = training_params.loss_fn
    accuracy_fn=training_params.accuracy_fn

    # Sequence padding function
    pad_sequence = utils.pad_sequence_with_empty_targets(
      generalization_task=task,
      computation_steps_mult=self._computation_steps_mult
    )

    steps = range(initial_step, training_params.training_steps)
    if self._use_tqdm:
      steps = tqdm.tqdm(steps)

    if initial_step < training_params.training_steps and not finished:

      """
      Begin training loop
      """

      # Frequency of logging training metrics (loss, accuracy)
      log_freq = training_params.log_frequency

      # Training steps
      for step in steps:

        model.train()

        if finished:
          break

        if step + 1 >= training_params.training_steps:
            finished = True

        # Sample sequence length according to curriculum
        length = length_curriculum.sample_sequence_length(step)

        # Sample a training batch
        train_batch = task.sample_batch(
            length=length, batch_size=training_params.batch_size)
        
        if training_params.use_query_token:
          train_batch['input'] = pad_sequence(train_batch['input'])

        train_batch_input = train_batch['input'].to(device)
        train_batch_output = train_batch['output'].to(device)

        optimizer.zero_grad()

        output = model(train_batch_input)

        train_loss, train_metrics = loss_fn(output, train_batch_output)

        if accuracy_fn is not None:
          train_accuracy = accuracy_fn(output, train_batch_output)
        else:
          train_accuracy = None

        train_loss.backward()

        # Gradient clipping
        if training_params.max_grad_norm > 0:
          nn.utils.clip_grad_norm_(model.parameters(), training_params.max_grad_norm)

        optimizer.step()

        # Logging the training loss/accuracy
        if (log_freq > 0) and (((step + 1) % log_freq == 0)):

          log_data = {
              "step": step,
              "train_loss": float(train_loss),
          }

          if training_params.accuracy_fn is not None:
            log_data["train_accuracy"] = float(train_accuracy)
          for key, value in train_metrics.items():
            log_data[".".join(["train_metrics", key])] = np.array(value)

          # Log to TB
          if self._training_params.tboard_logdir is not None:
            self.writer.add_scalar("Loss/train", float(train_loss), step)
            self.writer.add_scalar("Accuracy/train", float(train_accuracy), step)

          results.append(log_data)

          # Remove checkpoint for previous step
          for filename in sorted(os.listdir(save_path)):
            if 'step_' in filename:
                os.remove(os.path.join(save_path, filename))

          # Save checkpoint for current step
          t.save({
              'step': step+1,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              }, os.path.join(save_path, f'step_{step+1}.pt'))

    # Training loop is now finished
    if not finished:
      os.system(f'touch {os.path.join(save_path, "fin")}')

    # Evaluate the saved models
    for filename in sorted(os.listdir(save_path)):
        
        print(filename)

        if filename.endswith('best.pt') or 'step' in filename:
          checkpoint_name = filename

          checkpoint = t.load(os.path.join(save_path, checkpoint_name))
          model.load_state_dict(checkpoint['model_state_dict'])

          eval_params = range_evaluation.EvaluationParams(
              model=model,
              accuracy_fn=training_params.accuracy_fn,
              sample_batch=task.sample_batch,
              max_test_length=training_params.max_range_test_length,
              total_batch_size=training_params.range_test_total_batch_size,
              sub_batch_size=training_params.range_test_sub_batch_size,
              task=task,
              computation_steps_mult=self._computation_steps_mult,
              single_output=training_params.single_output,
              use_query_token=training_params.use_query_token,
          )

          eval_results = range_evaluation.range_evaluation(
            eval_params, use_tqdm=False, tboard_writer=self.writer)
                
    if self._training_params.tboard_logdir is not None:
      self.writer.flush()
      self.writer.close()

    return results, eval_results
