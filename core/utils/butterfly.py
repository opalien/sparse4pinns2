from torch import Tensor
import torch
import math
from einops import rearrange
from einops import einsum # type: ignore
from torch.nn import functional as F
import numpy as np

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





#class BlockdiagButterflyMultiply(torch.autograd.Function):
#
#    """This is a faster implementation, with careful memory copies for the fastest
#    bmm performance.
#    The backward pass is also written manually with careful memory copies.
#    Arguments:
#        x: (batch, n)
#        w1_bfly: (k, q, p), where k = n / p
#        w2_bfly: (l, s, r), where l = k * q / r = n * q / (p * r)
#    Outputs:
#        out: (batch, m), where m = l * s = n * s * q / (p * r)
#    """
#
#    @staticmethod
#    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
#    def forward(ctx, x, w1_bfly, w2_bfly):
#        batch_shape, n = x.shape[:-1], x.shape[-1]
#        batch_dim = np.prod(batch_shape)
#        k, q, p = w1_bfly.shape
#        l, s, r = w2_bfly.shape
#        assert k * p == n
#        assert l * r == k * q
#        x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
#        out1 = torch.empty(batch_dim, k, q, device=x.device, dtype=x.dtype).transpose(0, 1)
#        out1 = torch.bmm(x_reshaped, w1_bfly.transpose(-1, -2), out=out1)
#        out1 = out1.transpose(0, 1).reshape(batch_dim, r, l).transpose(-1, -2).contiguous().transpose(0, 1)
#        out2 = torch.empty(batch_dim, l, s, device=x.device, dtype=x.dtype).transpose(0, 1)
#        out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)
#        out2 = out2.permute(1, 2, 0).reshape(*batch_shape, s * l)
#        ctx.save_for_backward(x, w1_bfly, w2_bfly, out1)
#        return out2
#
#    @staticmethod
#    @torch.cuda.amp.custom_bwd
#    def backward(ctx, dout):
#        x, w1_bfly, w2_bfly, out1 = ctx.saved_tensors
#        batch_shape, n = x.shape[:-1], x.shape[-1]
#        batch_dim = np.prod(batch_shape)
#        k, q, p = w1_bfly.shape
#        l, s, r = w2_bfly.shape
#        # assert k * p == n
#        # assert l * r == k * q
#        dx, dw1_bfly, dw2_bfly = None, None, None
#        # dout_reshaped = dout.reshape(batch_dim, sqrtn, sqrtn).permute(2, 1, 0).contiguous()
#        dout_reshaped = dout.reshape(batch_dim, s, l).transpose(-1, -2).contiguous()
#        dout_reshaped = dout_reshaped.transpose(0, 1)
#        if ctx.needs_input_grad[2]:
#            # dw2_bfly = torch.empty(l, s, r, device=w2_bfly.device, dtype=w2_bfly.dtype)
#            # dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1, out=dw2_bfly)
#            dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1.conj())
#        if ctx.needs_input_grad[1] or ctx.needs_input_grad[0]:
#            dout1 = torch.empty(batch_dim, l, r, device=x.device, dtype=x.dtype).transpose(0, 1)
#            dout1 = torch.bmm(dout_reshaped, w2_bfly.conj())#, out=dout1)
#            dout1 = dout1.transpose(0, 1).transpose(-1, -2).contiguous().reshape(batch_dim, k, q).transpose(0, 1)
#            # dout1 = dout1.permute(1, 2, 0).contiguous().transpose(0, 1)
#            if ctx.needs_input_grad[0]:
#                dx = torch.empty(batch_dim, k, p, device=x.device, dtype=x.dtype)
#                # , out=dx.transpose(0, 1)
#                dx = torch.bmm(dout1, w1_bfly.conj()).transpose(0, 1).reshape(*batch_shape, n)
#            if ctx.needs_input_grad[1]:
#                x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
#                dw1_bfly = torch.bmm(dout1.transpose(-1, -2), x_reshaped.conj())
#        return dx, dw1_bfly, dw2_bfly
#

class BlockdiagButterflyMultiply(torch.autograd.Function):

    """This is a faster implementation, with careful memory copies for the fastest
    bmm performance.
    The backward pass is also written manually with careful memory copies.
    Arguments:
        x: (batch, n)
        w1_bfly: (k, q, p), where k = n / p
        w2_bfly: (l, s, r), where l = k * q / r = n * q / (p * r)
    Outputs:
        out: (batch, m), where m = l * s = n * s * q / (p * r)
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, w1_bfly, w2_bfly):
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        assert k * p == n
        assert l * r == k * q
        x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
        out1 = torch.empty(batch_dim, k, q, device=x.device, dtype=x.dtype).transpose(0, 1)
        out1 = torch.bmm(x_reshaped, w1_bfly.transpose(-1, -2), out=out1)
        out1 = out1.transpose(0, 1).reshape(batch_dim, r, l).transpose(-1, -2).contiguous().transpose(0, 1)
        out2 = torch.empty(batch_dim, l, s, device=x.device, dtype=x.dtype).transpose(0, 1)
        out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)
        out2 = out2.permute(1, 2, 0).reshape(*batch_shape, s * l)
        ctx.save_for_backward(x, w1_bfly, w2_bfly, out1)
        return out2

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout):
        x, w1_bfly, w2_bfly, out1 = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        # assert k * p == n
        # assert l * r == k * q
        dx, dw1_bfly, dw2_bfly = None, None, None
        # dout_reshaped = dout.reshape(batch_dim, sqrtn, sqrtn).permute(2, 1, 0).contiguous()
        dout_reshaped = dout.reshape(batch_dim, s, l).transpose(-1, -2).contiguous()
        dout_reshaped = dout_reshaped.transpose(0, 1)
        if ctx.needs_input_grad[2]:
            # dw2_bfly = torch.empty(l, s, r, device=w2_bfly.device, dtype=w2_bfly.dtype)
            # dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1, out=dw2_bfly)
            dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1.conj())
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[0]:
            dout1 = torch.empty(batch_dim, l, r, device=x.device, dtype=x.dtype).transpose(0, 1)
            dout1 = torch.bmm(dout_reshaped, w2_bfly.conj(), out=dout1)
            dout1 = dout1.transpose(0, 1).transpose(-1, -2).contiguous().reshape(batch_dim, k, q).transpose(0, 1)
            # dout1 = dout1.permute(1, 2, 0).contiguous().transpose(0, 1)
            if ctx.needs_input_grad[0]:
                dx = torch.empty(batch_dim, k, p, device=x.device, dtype=x.dtype)
                dx = torch.bmm(dout1, w1_bfly.conj(), out=dx.transpose(0, 1)).transpose(0, 1).reshape(*batch_shape, n)
            if ctx.needs_input_grad[1]:
                x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
                dw1_bfly = torch.bmm(dout1.transpose(-1, -2), x_reshaped.conj())
        return dx, dw1_bfly, dw2_bfly