# -*- coding: utf-8 -*-
"""Training schedulers.

This module contains classes implementing schedulers which control the
evolution of learning rule hyperparameters (such as learning rate) over a
training run.
"""

import numpy as np

class ConstantLearningRateScheduler(object):
    """Example of scheduler interface which sets a constant learning rate."""

    def __init__(self, learning_rate):
        """Construct a new constant learning rate scheduler object.

        Args:
            learning_rate: Learning rate to use in learning rule.
        """
        self.learning_rate = learning_rate

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Run at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        learning_rule.learning_rate = self.learning_rate
        
class ReciprocalLearningRateScheduler(object):
    
    def __init__(self, learning_rate, learning_rate_decay):
        """Construct a new constant learning rate scheduler object.

        Args:
            learning_rate: Learning rate to use in learning rule.
        """
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Run at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        """learning_rule.learning_rate = self.learning_rate*(math.exp(-epoch_number/50))
        """
        learning_rule.learning_rate = self.learning_rate/(1+epoch_number/self.learning_rate_decay)
        
class ExponentialLearningRateScheduler(object):
    
    def __init__(self, learning_rate, learning_rate_decay):
        """Construct a new constant learning rate scheduler object.

        Args:
            learning_rate: Learning rate to use in learning rule.
        """
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Run at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        """learning_rule.learning_rate = self.learning_rate*(math.exp(-epoch_number/50))
        """
        learning_rule.learning_rate = self.learning_rate*(np.exp(-(epoch_number/self.learning_rate_decay)))
        
class MomentumRateSchedule(object):
    
    def __init__(self, asymptoticm, speedup, speeddown):
        """Construct a new constant learning rate scheduler object.

        Args:
            learning_rate: Learning rate to use in learning rule.
        """
        self.asymptoticm = asymptoticm
        self.speedup = speedup
        self.speeddown = speeddown

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Run at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        """learning_rule.learning_rate = self.learning_rate*(math.exp(-epoch_number/50))
        """
        learning_rule.mom_coeff = self.asymptoticm *(1-self.speedup/(epoch_number+self.speeddown))