import torch
from torch import Tensor
import abc


TensorMeta = type(Tensor)

class TensorABCMeta(TensorMeta, abc.ABCMeta):
    pass


class Tensorable(abc.ABC):

    @property
    @abc.abstractmethod
    def dense(self) -> Tensor:
        raise NotImplementedError
    
    
    


class TensorLike(Tensorable, Tensor, abc.ABC, metaclass=TensorABCMeta):
    @abc.abstractmethod
    def to(self, device: torch.device | str) -> Tensor:
        raise NotImplementedError


    def __deepcopy__(self, memo: dict[int, "TensorLike"]) -> "TensorLike":
        raise NotImplementedError