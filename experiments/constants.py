#
# Copyright 2025 - IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Constants for our length generalization experiments."""

import functools
from State_Tracking_With_NNs.experiments import curriculum as curriculum_lib

"""
Modules
"""
from State_Tracking_With_NNs.models import rnn
from State_Tracking_With_NNs.models import lstm
from State_Tracking_With_NNs.models import SDSSM
from State_Tracking_With_NNs.models import SDSSM_nonlinear
from State_Tracking_With_NNs.models import complex_diagonal_no_B_linear
from State_Tracking_With_NNs.models import complex_diagonal_no_B_nonlinear
from State_Tracking_With_NNs.models import complex_diagonal_with_B_linear
from State_Tracking_With_NNs.models import complex_diagonal_with_B_nonlinear_multilayer


"""
Tasks
"""
from State_Tracking_With_NNs.tasks.regular import cycle_navigation
from State_Tracking_With_NNs.tasks.regular import even_pairs
from State_Tracking_With_NNs.tasks.regular import modular_arithmetic
from State_Tracking_With_NNs.tasks.regular import parity_check
from State_Tracking_With_NNs.tasks.regular import C_2_n
from State_Tracking_With_NNs.tasks.regular import A5
from State_Tracking_With_NNs.tasks.regular import D_n

MODEL_BUILDERS = {
    'rnn':
        rnn.ElmanRNN,
    'lstm':
        lstm.LSTM,
    'SDSSM':
        SDSSM.LSRNN,
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