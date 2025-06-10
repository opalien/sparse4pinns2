from torch import Tensor
from collections.abc import Callable

from core.datasets.pinn_dataset import PINNDataset

class AnyDataset(PINNDataset):
    def __init__(self, 
                 dirichlet_generator: Callable[[], tuple[Tensor, Tensor]], 
                 periodic_generator: Callable[[], tuple[Tensor, Tensor]], 
                 colloc_generator: Callable[[], Tensor],
                 n_dirichlet: int,
                 n_periodic: int,
                 n_colloc: int,
                 ):
        
        super().__init__() # type: ignore

        dirichlet = [dirichlet_generator() for _ in range(n_dirichlet)]
        periodic = [periodic_generator() for _ in range(n_periodic)]
        colloc = [colloc_generator() for _ in range(n_colloc)]
        self.set_dirichlet(dirichlet)
        self.set_periodic(periodic)
        self.set_colloc(colloc)


