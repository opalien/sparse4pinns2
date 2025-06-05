from core.models.pinn import PINN
from torch import Tensor
from torch import nn

import torch
import numpy as np
from numpy import ndarray as NDArray

from core.utils.train import train

class SimplePINN(PINN):

    def pde(self, u_pred: Tensor, a_in: Tensor, idx: int | None = None) -> Tensor:
        # Handle empty tensor case
        if u_pred.size(0) == 0:
            return u_pred.squeeze(-1)  # Return empty tensor with correct shape
        
        #J = self.J(a_in, u_pred)
        J = self.J()
        du_dt = J[:, 0, 0]
        du_dx = J[:, 0, 1]

        return du_dt + du_dx
    


from core.datasets.pinn_dataset import PINNDataset


class SimpleDataset(PINNDataset):
    def u(self, a: Tensor | NDArray[np.float64]) -> Tensor | NDArray[np.float64]:
        _, x = a[0], a[1:]
        match x:
            case _ if isinstance(x, Tensor):
                return np.cos(a[0]-a[1])#np.exp(-(np.pi/2)**2 * t) * torch.sin((np.pi/2)*(torch.tensor([x[0]])+1)) # type: ignore
            case _ if isinstance(x, np.ndarray):
                return np.cos(a[0]-a[1])
            case _:
                raise ValueError("x must be either a Tensor or a numpy array")



class TrainSimpleDataset(SimpleDataset):

    def __init__(self, n_elements: int, n_colloc: int):
        super().__init__()

        x_min = -1.
        x_max = 1.
        t_min = 0.
        t_max = 1.

        elements = [(
            a:=torch.tensor([0] + [np.random.uniform(x_min, x_max)]),
            torch.as_tensor([self.u(a)], dtype=torch.float32)
        ) for _ in range(n_elements)]

        elements.extend([(
            a:=torch.tensor([np.random.uniform(0, t_max),  x_min] ),
            torch.as_tensor([self.u(a)], dtype=torch.float32)
        ) for _ in range(n_elements)])

        elements.extend([(
            a:=torch.tensor([np.random.uniform(0, t_max),  x_max] ),
            torch.as_tensor([self.u(a)], dtype=torch.float32) 
        ) for _ in range(n_elements)])


        colloc = [torch.tensor([np.random.uniform(t_min, t_max)] 
                               + [np.random.uniform(x_min, x_max)]) 
                               for _ in range(n_colloc)]
        
        self.set_elements(elements)
        self.set_colloc(colloc)

        
        
class TestSimpleDataset(SimpleDataset):

    def __init__(self, n_elements: int):
        super().__init__()

        x_min = -1
        x_max = 1
        t_min = 0
        t_max = 1

        elements = [(
            a:=torch.tensor([np.random.uniform(t_min, t_max)] + [np.random.uniform(x_min, x_max)]),
            torch.as_tensor([self.u(a)], dtype=torch.float32)
        ) for _ in range(n_elements)]

        self.set_elements(elements)

        


if __name__ == "__main__":
    # Example usage

    n = 10
    k = 10

    layers = [
        nn.Linear(2, n),
        *[nn.Linear(n, n) for _ in range(k)],
        nn.Linear(n, 1),
    ]

    pinn = SimplePINN(layers)
    a_in = Tensor([[1.0, 2.0], [3.0, 4.0]])

    pinn(a_in)

    result = pinn.pde(pinn.u_pred, pinn.a_in, idx=1)
    print(result)


    train_dataset = TrainSimpleDataset(1000, 10_000)
    test_dataset  = TestSimpleDataset(100)

    train_dataloader  = train_dataset.get_dataloader(20)
    test_dataloader = test_dataset.get_dataloader(20)

    optimizer = torch.optim.Adam(pinn.parameters(), lr=0.001)

    train(pinn, train_dataloader, optimizer, torch.device("cpu"), 50, test_dataloader, verbose=True)