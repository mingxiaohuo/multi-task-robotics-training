import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        #return all the parameters with gradient
        params = sum(np.prod(p.size()) for p in model_parameters)
        #return the storage of all the parameters np.prod:caculate all the *
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
