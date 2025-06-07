import torch
from torch import Tensor
from typing import Literal



element_type = tuple[Literal[0],tuple[Tensor, Tensor]]
periodic_type = tuple[Literal[1], list[Tensor]]
colloc_type = tuple[Literal[2], Tensor]

TYPE = element_type | periodic_type | colloc_type



PINNDataloader = torch.utils.data.DataLoader[TYPE]

class PINNDataset(torch.utils.data.Dataset[TYPE]):
    def __init__(self):
        super().__init__()

        # Bounds conditions
        # u(elements[:][0]) = elements[:][1]
        self.elements: list[ tuple[Tensor, Tensor] ] = []

        # Periodic boundary conditions
        # size of elements of periodic can be different
        # u(periodic[i][j]) = u(periodic[i][k]), where j != k
        self.periodic: list[list[Tensor]] = []

        # Collocation points such PDE is respected
        # MSE(O(colloc[:])) = 0
        self.colloc: list[Tensor] = []


        self.u_zero: Tensor | None = None


    def set_elements(self, elements: list[ tuple[Tensor, Tensor] ]):
        self.elements = elements
        if len(self.elements) > 0:
            self.u_zero = torch.zeros_like(self.elements[0][1])


    def set_periodic(self, periodic: list[list[Tensor]]):
        self.periodic = periodic
    

    def set_colloc(self, colloc: list[Tensor]):
        self.colloc = colloc

    
    def append_element(self, a:Tensor, u:Tensor):
        a = torch.as_tensor(a, dtype=torch.float32)
        u = torch.as_tensor(u, dtype=torch.float32)
        self.elements.append((a, u))
        if self.u_zero is None:
            self.u_zero = torch.zeros_like(u)
            

    def append_periodic(self, A: list[Tensor]):
        A = [torch.as_tensor(a, dtype=torch.float32) for a in A]
        self.periodic.append(A)
    

    def append_colloc(self, colloc:Tensor):
        colloc = torch.as_tensor(colloc, dtype=torch.float32)
        self.colloc.append(colloc)




    def __len__(self):
        return len(self.elements) + len(self.periodic) + len(self.colloc)


    # idx -> [type, elements | colloc | periodic]
    def __getitem__(self, idx: int) -> TYPE:
        match idx:
            case _ if idx < len(self.elements):
                a, u = self.elements[idx]
                return 0, (a, u)
            
            case _ if idx < len(self.elements) + len(self.periodic):
                A = self.periodic[idx - len(self.elements)]
                return 1, A

            case _ if idx < len(self.elements) + len(self.periodic) + len(self.colloc):
                a = self.colloc[idx - len(self.elements) - len(self.periodic)]
                return 2, a

            case _:
                raise IndexError("Index out of range")


    def get_dataloader(self, batch_size: int, shuffle: bool = True) -> PINNDataloader:
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn) # type: ignore
    

def collate_fn(batch: list[TYPE]) -> tuple[tuple[Tensor, Tensor], list[list[Tensor]], Tensor]:

    batch_list: tuple[list[tuple[Tensor, Tensor]], list[list[Tensor]], list[Tensor]] = ([], [], [])

    for item in batch:
        batch_list[item[0]].append(item) # type: ignore

    elements: tuple[Tensor, Tensor] = (
        torch.stack([a for _, (a, _) in batch_list[0]]),
        torch.stack([u for _, (_, u) in batch_list[0]])
    )

    periodic = batch_list[1]

    colloc: Tensor = torch.stack(batch_list[2])

    return elements, periodic, colloc

