#
# Copyright 2025 - IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#




"""Example script to train and evaluate a network."""

import torch as t
import torch.nn.functional as F
import os
from absl import app
from absl import flags
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from State_Tracking_With_NNs.experiments import constants
from State_Tracking_With_NNs.experiments import curriculum as curriculum_lib
from State_Tracking_With_NNs.experiments import training

# General experiment flags
_EXPERIMENT_DIR = flags.DEFINE_string(
  'experiment_dir',
  default='results',
  help='main experiment directory'
)

_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size',
    default=128,
    help='Training batch size.',
    lower_bound=1,
)

_TRAIN_LENGTH = flags.DEFINE_integer(
    'train_length',
    default=40,
    help='Maximum training sequence length.',
    lower_bound=1,
)

_TEST_LENGTH = flags.DEFINE_integer(
    'test_length',
    default=500,
    help='Maximum test length'
)

_TASK = flags.DEFINE_string(
    'task',
    default='parity_check',
    help='Length generalization task (see `constants.py` for other tasks).',
)

_COMPUTATION_STEPS_MULT = flags.DEFINE_integer(
    'computation_steps_mult',
    default=0,
    help=(
        'The amount of computation tokens to append to the input tape (defined'
        ' as a multiple of the input length)'
    ),
    lower_bound=0,
)

_VALIDATION_FREQUENCY = flags.DEFINE_integer(
  'validation_frequency',
  default=20_000,
  help='validate model every n-many steps on in-domain lengths'
)

_TRAIN_STEPS = flags.DEFINE_integer(
  'train_steps',
  default=1_000_000,
  help='number of training steps.'
)

_USE_QUERY_TOKEN = flags.DEFINE_boolean(
  'use_query_token',
  default=False,
  help='query token on/off'
)

_SEED = flags.DEFINE_integer(
  'seed',
  default=0,
  help='random seed.'
)

# Optimizer flags
_OPTIMIZER = flags.DEFINE_string(
  'optimizer',
  default='Adam',
  help='choose the optimizer'
)

_MOMENTUM = flags.DEFINE_float(
  'momentum',
  default=0,
  help='SGD momentum'
)

_DAMPENING = flags.DEFINE_float(
  'dampening',
  default=0,
  help='SGD dampening'
)

_LEARNING_RATE = flags.DEFINE_float(
  'lr',
  default=1e-4,
  help='learning rate'
)

_GRAD_CLIP = flags.DEFINE_float(
  'grad_clip',
  default=1,
  help='gradient clipping threshold'
)

_WEIGHT_DECAY = flags.DEFINE_float(
  'wd',
  default=0.0,
  help='l2 weight penalty.'
)

# Architectural flags
_ARCHITECTURE = flags.DEFINE_string(
    'architecture',
    default='rnn',
    help='Model architecture (see `constants.py` for other architectures).',
)

_NUM_LAYERS = flags.DEFINE_integer(
  'num_layers',
  default=1,
  help='number of layers'
)

_EMBED_SIZE = flags.DEFINE_integer(
  'embed_size',
  default=-1,
  help='embedding size'
)

_STATE_SIZE = flags.DEFINE_integer(
  'state_size',
  default=256,
  help='linear models state size'
)

_MLP_SIZE = flags.DEFINE_integer(
  'mlp_size',
  default=64,
  help='mlp readout hidden size'
)

_NUM_TRANSITION_MATRICES = flags.DEFINE_integer(
  'num_transition_matrices',
  default=2,
  help='number of A matrices in the selective linear RNN'
)

_NONLINEARITY = flags.DEFINE_string(
  'nonlinearity',
  default='relu',
  help=''
)

_LP_NORM = flags.DEFINE_float(
  'Lp_norm',
  default=1,
  help='p in Lp norm when normalizing columns of matrices'
)

# Logging flags
_LOG_FREQUENCY = flags.DEFINE_integer(
  'log_frequency',
  default=5_000,
  help='interval at which values are logged and models are checkpointed'
)

_SPECTRUM_LOG_FREQUENCY = flags.DEFINE_integer(
  'spectrum_log_frequency',
  default=10_000,
  help='interval at which logged spectrum-related values are logged'
)

# The architecture parameters depend on the architecture, so we cannot define
# them as via flags. See `constants.py` for the required values.
_ARCHITECTURE_PARAMS = {
   "norm_type": "layernorm",
   "mlp_size_mult": 1,
   "prenorm": True
 }

def main(unused_argv) -> None:
  
  # Create the task.
  # Functionality: Random number samples
  curriculum = curriculum_lib.UniformCurriculum(
      values=list(range(1, _TRAIN_LENGTH.value + 1))
  )
  # Bit of a hack
  n = None
  if _TASK.value in ['C2xC30', 'D30']:
    n = 30
  elif _TASK.value in ['C2xC4', 'D4']:
    n = 4

  # Creates the task object
  if n is None:
    task = constants.TASK_BUILDERS[_TASK.value]()
  else:
    task = constants.TASK_BUILDERS[_TASK.value](n=n)

  print(f"Task: {_TASK.value}")

  # Create the model.
  single_output = task.output_length(10) == 1

  _ARCHITECTURE_PARAMS['num_layers'] = _NUM_LAYERS.value
  _ARCHITECTURE_PARAMS['num_transition_matrices'] = _NUM_TRANSITION_MATRICES.value
  _ARCHITECTURE_PARAMS['nonlinearity'] = _NONLINEARITY.value
  _ARCHITECTURE_PARAMS['max_seq_len'] = _TEST_LENGTH.value

  # Parameters for the general linear RNN
  _ARCHITECTURE_PARAMS['state_size'] = _STATE_SIZE.value
  _ARCHITECTURE_PARAMS['Lp_norm'] = _LP_NORM.value
  _ARCHITECTURE_PARAMS['mlp_size'] = _MLP_SIZE.value
  
  max_test_length = _TEST_LENGTH.value

  # Optional additional "query" token
  if _USE_QUERY_TOKEN.value:
    extra_dims_onehot = 1 + int(_COMPUTATION_STEPS_MULT.value > 0)
  else:
    extra_dims_onehot = 0
  final_input_size = task.input_size + extra_dims_onehot

  if _EMBED_SIZE.value == -1:
    embed_size = final_input_size
  
  model = constants.MODEL_BUILDERS[_ARCHITECTURE.value](
      output_size=task.output_size,
      return_all_outputs=not single_output,
      input_size=final_input_size,
      embed_size=embed_size,
      task=_TASK.value,
      **_ARCHITECTURE_PARAMS,
  )

  print(model)

  # Result paths
  modelname = f"{_ARCHITECTURE.value}_{_TASK.value}_layers_{_ARCHITECTURE_PARAMS['num_layers']}_embed_{_EMBED_SIZE.value}_state_{_ARCHITECTURE_PARAMS['state_size']}_k_{_ARCHITECTURE_PARAMS['num_transition_matrices']}_opt_{_OPTIMIZER.value}_lr_{_LEARNING_RATE.value}_trainlen_{_TRAIN_LENGTH.value}"
  
  if _GRAD_CLIP.value !=1:
    modelname += f'_gradclip_{_GRAD_CLIP.value}'

  if _LP_NORM.value != 1:
    modelname += f'_lpnorm_{_LP_NORM.value}'

  if _WEIGHT_DECAY.value != 0:
    modelname += f'wd_{_WEIGHT_DECAY.value}'

  if _MOMENTUM.value != 0:
    modelname += f'mom_{_MOMENTUM.value}'

  if _DAMPENING.value != 0:
    modelname += f'damp_{_DAMPENING.value}'


  if _ARCHITECTURE.value == 'SDSSM_MLP_ReLU':
    modelname += f'mlpsize_{_MLP_SIZE.value}'
      
  modelname += f'_SEED_{_SEED.value}'

  savedir = f"{_EXPERIMENT_DIR.value}/{_TASK.value}/{_ARCHITECTURE.value}/{modelname}"
  
  os.makedirs(savedir, exist_ok=True)

  # Create the loss and accuracy based on the pointwise ones.
  def loss_fn(output, target):
    loss = t.mean(t.sum(task.pointwise_loss_fn(output, target), axis=-1))
    return loss, {}

  def accuracy_fn(output, target):
    mask = task.accuracy_mask(target)
    return t.sum(mask * task.accuracy_fn(output, target)) / t.sum(mask)

  # Create the final training parameters.
  training_params = training.ClassicTrainingParams(
      seed=_SEED.value,
      architecture=_ARCHITECTURE.value,
      training_steps=_TRAIN_STEPS.value,
      log_frequency=_LOG_FREQUENCY.value, 
      spectrum_log_frequency = _SPECTRUM_LOG_FREQUENCY.value,
      length_curriculum=curriculum,
      batch_size=_BATCH_SIZE.value,
      task=task,
      model=model,
      loss_fn=loss_fn,
      learning_rate=_LEARNING_RATE.value,
      weight_decay=_WEIGHT_DECAY.value,
      accuracy_fn=accuracy_fn,
      max_range_test_length=max_test_length,
      range_test_total_batch_size=512, # What is this
      range_test_sub_batch_size=64, # What is this
      single_output=single_output,
      tboard_logdir=os.path.join(savedir, 'tensorboard_log'),
      save_path = savedir,
      max_grad_norm = _GRAD_CLIP.value,
      validation_frequency = _VALIDATION_FREQUENCY.value,
      optimizer = _OPTIMIZER.value,
      state_size=_STATE_SIZE.value,
      embed_size=embed_size,
      num_transition_matrices=_NUM_TRANSITION_MATRICES.value,
      train_length=_TRAIN_LENGTH.value)

  # This is the main training class
  training_worker = training.TrainingWorker(training_params, 
                                            use_tqdm=True,
                                            computation_steps_mult=_COMPUTATION_STEPS_MULT.value)

  train_results, eval_results_all, eval_lengths = training_worker.run()

  print(f'Train results: {train_results}')
  
  for i, eval_results in enumerate(eval_results_all):
    os.makedirs(os.path.join(savedir, f'evaluation_len_{eval_lengths[i]}'), exist_ok=True)
    result_dict = {}
    for entry in eval_results:
      result_dict[entry['length']] = entry['accuracy']

    json_dict_object = json.dumps(result_dict, indent=4)

    with open(os.path.join(savedir, f'evaluation_len_{eval_lengths[i]}', 'results_dict.json'), 'w') as outfile:
      outfile.write(json_dict_object)
  
if __name__ == '__main__':
  app.run(main)