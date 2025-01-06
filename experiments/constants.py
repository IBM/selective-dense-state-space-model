#
# Copyright 2025 - IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""
Constants for our length generalization experiments.
"""
from selective_dense_state_space_model.experiments import curriculum as curriculum_lib

"""
Modules
"""
from selective_dense_state_space_model.models import rnn
from selective_dense_state_space_model.models import lstm
from selective_dense_state_space_model.models import SDSSM
from selective_dense_state_space_model.models import SDSSM_nonlinear
from selective_dense_state_space_model.models import complex_diagonal_no_B_linear
from selective_dense_state_space_model.models import complex_diagonal_no_B_nonlinear
from selective_dense_state_space_model.models import complex_diagonal_with_B_linear
from selective_dense_state_space_model.models import complex_diagonal_with_B_nonlinear_multilayer

"""
Tasks
"""
from selective_dense_state_space_model.tasks.regular import cycle_navigation
from selective_dense_state_space_model.tasks.regular import even_pairs
from selective_dense_state_space_model.tasks.regular import modular_arithmetic
from selective_dense_state_space_model.tasks.regular import parity_check
from selective_dense_state_space_model.tasks.regular import C_2_n
from selective_dense_state_space_model.tasks.regular import A5
from selective_dense_state_space_model.tasks.regular import D_n

MODEL_BUILDERS = {
    'rnn':
        rnn.ElmanRNN,
    'lstm':
        lstm.LSTM,
    'SDSSM':
        SDSSM.SDSSM,
    'SDSSM_nonlinear':
        SDSSM_nonlinear.LSRNN,
    'complex_diagonal_with_B_linear':
        complex_diagonal_with_B_linear.LSRNN,
    'complex_diagonal_with_B_nonlinear':
        complex_diagonal_with_B_nonlinear_multilayer.LSRNN,
    'complex_diagonal_no_B_linear':
        complex_diagonal_no_B_linear.LSRNN,
    'complex_diagonal_no_B_nonlinear':
        complex_diagonal_no_B_nonlinear.LSRNN,
}

CURRICULUM_BUILDERS = {
    'fixed': curriculum_lib.FixedCurriculum,
    'regular_increase': curriculum_lib.RegularIncreaseCurriculum,
    'reverse_exponential': curriculum_lib.ReverseExponentialCurriculum,
    'uniform': curriculum_lib.UniformCurriculum,
}

TASK_BUILDERS = {
    'C2xC30':
        C_2_n.C_2_n,
    'C2xC4':
        C_2_n.C_2_n,
    'D4':
        D_n.DihedronNavigation,
    'D30':
        D_n.DihedronNavigation,
    'A5':
        A5.A5Navigation,
    'modular_arithmetic':
        modular_arithmetic.ModularArithmetic,
    'parity_check':
        parity_check.ParityCheck,
    'even_pairs':
        even_pairs.EvenPairs,
    'cycle_navigation':
        cycle_navigation.CycleNavigation,
}