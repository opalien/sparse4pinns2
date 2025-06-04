import torch
from torch import Tensor

PINNDataloader = torch.utils.data.DataLoader[tuple[Tensor, Tensor, bool]]

class PINNDataset(torch.utils.data.Dataset[tuple[Tensor, Tensor, bool]]):
    def __init__(self):
        self.elements: list[ tuple[Tensor, Tensor] ] = []
        self.colloc: list[Tensor] = []
        self.u_zero: Tensor | None = None


    def set_elements(self, elements: list[ tuple[Tensor, Tensor] ]):
        self.elements = elements
        if len(self.elements) > 0:
            self.u_zero = torch.zeros_like(self.elements[0][1])

    
    def set_colloc(self, colloc: list[Tensor]):
        self.colloc = colloc

    
    def append_element(self, a:Tensor, u:Tensor):
        a = torch.as_tensor(a, dtype=torch.float32)
        u = torch.as_tensor(u, dtype=torch.float32)
        self.elements.append((a, u))
        if self.u_zero is None:
            self.u_zero = torch.zeros_like(u)
        
    
    def append_colloc(self, colloc:Tensor):
        colloc = torch.as_tensor(colloc, dtype=torch.float32)
        self.colloc.append(colloc)


    def __len__(self):
        return len(self.elements) + len(self.colloc)
    

    def __getitem__(self, idx: int)-> tuple[Tensor, Tensor, bool]:
        match idx < len(self.elements):
            case True:
                a, u = self.elements[idx]
                return a, u, True
            
            case False:
                if self.u_zero is None:
                    raise ValueError("u_zero is None, please set it before using the dataset")
                a = self.colloc[idx - len(self.elements)]
                return a, self.u_zero, False


    def get_dataloader(self, batch_size: int, shuffle: bool = True) -> PINNDataloader:
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn) # type: ignore
    

def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, bool]]):
    e_a: list[Tensor] = []
    e_u: list[Tensor] = []

    c_a: list[Tensor] = []
    c_u: list[Tensor] = []

    for a, u, is_e in batch:
        if is_e:
            e_a.append(a)
            e_u.append(u)
        else:
            c_a.append(a)
            c_u.append(u)

    idx = len(e_a)

    a = torch.stack(e_a + c_a)
    u = torch.stack(e_u + c_u)

    return a, u, idx