from torch import Tensor
import torch
import math
from einops import rearrange
from einops import einsum # type: ignore



def low_rank_project(M: Tensor) -> tuple[Tensor, Tensor]:
    rank = 1
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)

    S_sqrt = torch.sqrt(S[..., :rank]).to(M.device)

    U_proj = U[..., :, :rank] * rearrange(S_sqrt, '... rank -> ... 1 rank')
    Vh_proj = rearrange(S_sqrt, '... rank -> ... rank 1') * Vh[..., :rank, :]
    return U_proj, Vh_proj


def factors(n: int):
    limit = math.floor(math.sqrt(n)) + 1
    return [(i, n // i) for i in range(1, limit) if n % i == 0]


def blockdiag_butterfly_project(M:Tensor, sizes: tuple[int, int] | None = None):
    m, n = M.shape
    if m != n:
        raise NotImplementedError('Only support square matrices')
    if sizes is None:
        size_factors = factors(n)
        if not size_factors: 
             raise ValueError(f"Could not find factors for n={n}")
        closest_factors = size_factors[-1]
        sizes = (closest_factors[1], closest_factors[0])
        
    assert n == sizes[0] * sizes[1], f"Product of sizes {sizes[0]}*{sizes[1]} does not equal matrix dimension {n}"
    
    M_permuted_batched = rearrange(M, '(p k) (r s) -> k r p s', k=sizes[1], r=sizes[0])
    
    U, Vh = low_rank_project(M_permuted_batched)
    
    w1_bfly = rearrange(Vh, 'k r 1 s -> r k s')
    w2_bfly = rearrange(U, 'k r s 1 -> k s r')
    
    return w1_bfly, w2_bfly