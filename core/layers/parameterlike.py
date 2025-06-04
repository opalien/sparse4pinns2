import torch
from torch import nn
import abc
from ..tensors.tensorlike import Tensorable


class ParameterLike(Tensorable, nn.Module, abc.ABC):
    #@abc.abstractmethod
    #def to(self, device: torch.device | str) -> nn.Module:
    #    raise NotImplementedError
    pass

