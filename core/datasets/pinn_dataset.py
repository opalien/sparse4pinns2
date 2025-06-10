import torch
from torch import Tensor
from typing import Literal



dirichlet_type = tuple[Tensor, Tensor]
periodic_type = tuple[Tensor, Tensor]
colloc_type = Tensor

TYPE = tuple[Literal[0], dirichlet_type] | tuple[Literal[1], periodic_type] | tuple[Literal[2], colloc_type]


PINNDataloader = torch.utils.data.DataLoader[TYPE]

class PINNDataset(torch.utils.data.Dataset[TYPE]):
    def __init__(self):
        super().__init__()

        # Bounds conditions
        # u(dirichlet[:][0]) = dirichlet[:][1]
        self.dirichlet: list[dirichlet_type] = []

        # Periodic boundary conditions
        # size of dirichlet of periodic can be different
        # u(periodic[:][0]) = u(periodic[:][1])
        self.periodic: list[periodic_type] = []

        # Collocation points such PDE is respected
        # MSE(O(colloc[:])) = 0
        self.colloc: list[colloc_type] = []


        self.u_zero: Tensor | None = None


    def set_dirichlet(self, dirichlet: list[ dirichlet_type ]):
        self.dirichlet = dirichlet
        if len(self.dirichlet) > 0:
            self.u_zero = torch.zeros_like(self.dirichlet[0][1])


    def set_periodic(self, periodic: list[ periodic_type ]):
        self.periodic = periodic
    

    def set_colloc(self, colloc: list[ colloc_type ]):
        self.colloc = colloc

    
    def append_dirichlet(self, a: Tensor, u: Tensor):
        a = torch.as_tensor(a, dtype=torch.float32)
        u = torch.as_tensor(u, dtype=torch.float32)
        self.dirichlet.append((a, u))
        if self.u_zero is None:
            self.u_zero = torch.zeros_like(u)
            

    def append_periodic(self, a_1: Tensor, a_2: Tensor):
        a_1 = torch.as_tensor(a_1, dtype=torch.float32)
        a_2 = torch.as_tensor(a_2, dtype=torch.float32)
        self.periodic.append((a_1, a_2))


    def append_colloc(self, colloc: Tensor):
        colloc = torch.as_tensor(colloc, dtype=torch.float32)
        self.colloc.append(colloc)




    def __len__(self):
        return len(self.dirichlet) + len(self.periodic) + len(self.colloc)


    # idx -> [type, dirichlet | colloc | periodic]
    def __getitem__(self, idx: int) -> TYPE:
        match idx:
            case _ if idx < len(self.dirichlet):
                a, u = self.dirichlet[idx]
                return 0, (a, u)
            
            case _ if idx < len(self.dirichlet) + len(self.periodic):
                A = self.periodic[idx - len(self.dirichlet)]
                return 1, A

            case _ if idx < len(self.dirichlet) + len(self.periodic) + len(self.colloc):
                a = self.colloc[idx - len(self.dirichlet) - len(self.periodic)]
                return 2, a

            case _:
                raise IndexError("Index out of range")


    def get_dataloader(self, batch_size: int, shuffle: bool = True) -> PINNDataloader:
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn) # type: ignore
    

def collate_fn(batch: list[TYPE]) -> tuple[ dirichlet_type, periodic_type, colloc_type ]:

    batch_list: tuple[list[dirichlet_type], list[periodic_type], list[colloc_type]] = ([], [], [])

    for item in batch:
        batch_list[item[0]].append(item[1]) # type: ignore

    #print(f"{batch_list[:10]=}")

    dirichlet: dirichlet_type = (torch.tensor([]), torch.tensor([]))
    if len(batch_list[0]) > 0:
        dirichlet: dirichlet_type = (
            torch.stack([a for (a, _) in batch_list[0]]),
            torch.stack([u for (_, u) in batch_list[0]])
        )

    periodic: periodic_type = (torch.tensor([]), torch.tensor([]))
    if len(batch_list[1]) > 0:
        periodic: periodic_type = (
            (torch.stack([a_1 for (a_1, _) in batch_list[1]]),
            torch.stack([a_2 for (_, a_2) in batch_list[1]]))
        )

    colloc: colloc_type = torch.tensor([])
    if len(batch_list[2]) > 0:
        colloc: colloc_type = torch.stack(batch_list[2])

    return dirichlet, periodic, colloc

