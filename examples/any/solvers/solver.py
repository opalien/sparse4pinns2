import torch
from torch import Tensor

import abc


class Solver(abc.ABC):
    
    @abc.abstractmethod
    def solve(self, *args, **kwargs) -> None:
        raise NotImplementedError
    
    @abc.abstractmethod
    def func(self, a: Tensor) -> Tensor:
        raise NotImplementedError


