"""FFN layer boundary that recomputes the attention residual (kernels/boundary_ffn_remat.cu).

The FFN-side pair of boundary_attn_remat.py: the attention boundary wrote only the FFN input and each row's
post-norm scale, and this kernel rebuilds the residual after attention from the layer's input stream and the
attention branch before applying the FFN boundary, so the mid-layer FP32 residual is never written. layer.py uses it
on the 2048+ token path when supported() holds for the two boundaries' parameters; with the token order it also takes
the inverse map and returns the top-8 ids in token order.
"""
import torch
from torch.utils.cpp_extension import load_inline
from cuda_source import cuda_source

_SRC = cuda_source("boundary_ffn_remat.cu")

_nr = load_inline(name='okc_remat_z08020', cuda_sources=_SRC,
    cpp_sources='void remat_init(); void remat_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,double,double,int64_t,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);',
    functions=['remat_init','remat_run'],
    extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v'],verbose=False)

GRID = 132  # one persistent CTA per H100 SM
R2 = True  # remat2_k (parameters in shared memory); False: remat_k (TMA-staged rows)
_initialized = False

def init():
    global _initialized
    if not _initialized:
        _nr.remat_init()
        _initialized = True

def supported(pa,pb):
    # The kernel's fixed form: attention post-norm without scale or weight, FFN post-norm with input scale, width 8192.
    return (not pa['scale'] and pa['postw'].numel()==0 and pb['scale']==1
            and all(p[k].numel()==8192 for p in (pa,pb) for k in ('alpha','beta','prew'))
            and pb['postw'].numel()==8192)

_E = {}

def _empty(dev):
    if dev not in _E:
        _E[dev] = torch.empty(0, dtype=torch.int32, device=dev)
    return _E[dev]

def run_b(pa,pb,resid,attn_out,ffn_out,rps,out,new,grid=132,inv=None,ids_in=None,ids_out=None):
    init()
    e=_empty(resid.device)
    _nr.remat_run(resid,attn_out,ffn_out,out,new,rps,pa['alpha'],pa['beta'],pb['alpha'],pb['beta'],
                  pb['postw'],pb['prew'],pb['post_eps'],pb['pre_eps'],grid,
                  e if inv is None else inv,e if inv is None else ids_in,e if inv is None else ids_out,1 if R2 else 0)

def boundary_b(pa,pb,resid,attn_out,ffn_out,rps,inv=None,ids=None):
    """(next input, new residual); with the inverse map inv (token t's FFN row, row scale and ids at row inv[t]) also the
    top-8 ids in token order."""
    out,new=torch.empty_like(ffn_out),torch.empty_like(resid)
    if inv is None:
        run_b(pa,pb,resid,attn_out,ffn_out,rps,out,new,GRID)
        return out,new
    ids_out=torch.empty_like(ids)
    run_b(pa,pb,resid,attn_out,ffn_out,rps,out,new,GRID,inv,ids,ids_out)
    return out,new,ids_out
