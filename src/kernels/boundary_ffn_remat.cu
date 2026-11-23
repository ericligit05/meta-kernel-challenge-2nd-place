// The FFN layer boundary with the attention residual rematerialized (the pair of boundary_attn_remat.cu's
// attention side): per token, x = alpha_a * stream + beta_a * bf16(attn * rps) recomputes the residual after
// attention from the attention branch and its stored post-norm row scale rps, then the FFN boundary
// (post-norm with input scale, gated add, pre-norm) writes the new FP32 residual and the next BF16 input.
// Both kernels are persistent (grid CTAs of 1024 threads, CTA c walks tokens c, c + grid, ...). With MAP, token t's
// FFN branch row, row scale and top-8 ids sit at row inv[t] (the token order), and the ids are copied back to token
// order. remat2_k is the one in use (boundary_ffn_remat.py R2); remat_k is an unused TMA-staged form of the same math.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

namespace {
constexpr int RMAX = 128;  // tokens per CTA with a map (T <= RMAX x grid)
constexpr int NT = 1024, VEC = 8192 / NT, GROUPS = VEC / 4, WARPS = NT / 32;
__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ void ld4(const __nv_bfloat16* p, int i, float* f) {
  uint2 v = reinterpret_cast<const uint2*>(p)[i];
  float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
  float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
  f[0]=a.x; f[1]=a.y; f[2]=b.x; f[3]=b.y;
}
// Four BF16 (two packed words) to float, as shift / mask in one asm block.
__device__ __forceinline__ void reg4(uint32_t lo, uint32_t hi, float* f) {
  uint32_t a,b,c,d;
  asm volatile("{ .reg .b32 x,y,z,w; "
      "shl.b32 x, %4, 16; and.b32 y, %4, 0xffff0000; "
      "shl.b32 z, %5, 16; and.b32 w, %5, 0xffff0000; "
      "mov.b32 %0, x; mov.b32 %1, y; mov.b32 %2, z; mov.b32 %3, w; }"
      : "=r"(a),"=r"(b),"=r"(c),"=r"(d) : "r"(lo),"r"(hi));
  f[0]=__uint_as_float(a); f[1]=__uint_as_float(b);
  f[2]=__uint_as_float(c); f[3]=__uint_as_float(d);
}
__device__ __forceinline__ float bsum(float v, float* red) {
  #pragma unroll
  for (int o=16; o>0; o>>=1) v+=__shfl_xor_sync(0xffffffffu,v,o);
  if ((threadIdx.x&31)==0) red[threadIdx.x/32]=v;
  __syncthreads();
  float s=red[(threadIdx.x & 31) & (WARPS - 1)];
  #pragma unroll
  for (int o=WARPS/2; o>0; o>>=1) s+=__shfl_xor_sync(0xffffffffu,s,o);
  return s;
}
__device__ __forceinline__ void minit(uint32_t a) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(a) : "memory");
}
__device__ __forceinline__ void expect(uint32_t a) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 65536;" :: "r"(a) : "memory");
}
__device__ __forceinline__ void wait(uint32_t a, int p) {
  asm volatile("{ .reg .pred q; W: mbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1; @!q bra W; }"
      :: "r"(a),"r"(p) : "memory");
}
__device__ __forceinline__ void bulk(uint32_t d, const void* s, int n, uint32_t b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
      :: "r"(d),"l"(s),"r"(n),"r"(b) : "memory");
}

// remat_k: per-column parameters in registers; the next token's three rows (FP32 stream 32 KB, BF16 attention and
// FFN branches 16 KB each) are staged by cp.async.bulk into a two-slot 64 KB shared ring with one mbarrier per slot.
template <bool MAP>
__global__ void __launch_bounds__(NT,1) remat_k(const float* __restrict__ resid,
    const __nv_bfloat16* __restrict__ attn, const __nv_bfloat16* __restrict__ ffn,
    __nv_bfloat16* __restrict__ out, float* __restrict__ nst, const float* __restrict__ rps,
    const float* __restrict__ aa, const float* __restrict__ ba,
    const float* __restrict__ ab, const float* __restrict__ bb,
    const __nv_bfloat16* __restrict__ postw, const __nv_bfloat16* __restrict__ prew,
    float post_eps, float pre_eps, int T,
    const int* __restrict__ inv, const int* __restrict__ ids_in, int* __restrict__ ids_out) {
  __shared__ float red[2][WARPS];
  __shared__ __align__(16) uint64_t bars[2];
  extern __shared__ __align__(128) unsigned char stage[];
  __shared__ int prow[MAP ? RMAX : 1];    // inv of this CTA's tokens blockIdx.x + k * gridDim.x
  __shared__ float prps[MAP ? RMAX : 1];  // their row scales
  const int nk=(int)((T-1-(int)blockIdx.x)/(int)gridDim.x)+1;  // tokens of this CTA
  if (MAP)  // the map is complete before nr3_k ran (full stream wait): safe to read before the wait
    for (int k=threadIdx.x; k<nk; k+=NT) prow[k]=__ldg(inv+blockIdx.x+(size_t)k*gridDim.x);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int tid=threadIdx.x;
  float ar[VEC], br[VEC], cr[VEC], dr[VEC];
  uint32_t pre_r[2 * GROUPS], post_r[2 * GROUPS];
  #pragma unroll
  for (int g=0; g<GROUPS; ++g) {
    float4 a=reinterpret_cast<const float4*>(aa)[NT*g+tid];
    float4 b=reinterpret_cast<const float4*>(ba)[NT*g+tid];
    float4 c=reinterpret_cast<const float4*>(ab)[NT*g+tid];
    float4 d=reinterpret_cast<const float4*>(bb)[NT*g+tid];
    ar[4*g]=a.x; ar[4*g+1]=a.y; ar[4*g+2]=a.z; ar[4*g+3]=a.w;
    br[4*g]=b.x; br[4*g+1]=b.y; br[4*g+2]=b.z; br[4*g+3]=b.w;
    cr[4*g]=c.x; cr[4*g+1]=c.y; cr[4*g+2]=c.z; cr[4*g+3]=c.w;
    dr[4*g]=d.x; dr[4*g+1]=d.y; dr[4*g+2]=d.z; dr[4*g+3]=d.w;
    uint2 p=reinterpret_cast<const uint2*>(prew)[NT*g+tid];
    uint2 q=reinterpret_cast<const uint2*>(postw)[NT*g+tid];
    pre_r[2*g]=p.x; pre_r[2*g+1]=p.y;
    post_r[2*g]=q.x; post_r[2*g+1]=q.y;
  }
  const uint32_t sa=(uint32_t)__cvta_generic_to_shared(stage);
  const uint32_t ma=(uint32_t)__cvta_generic_to_shared(bars);
  if (tid==0) {
    minit(ma); minit(ma+8);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  auto prefetch=[&](size_t row,int slot,int k) {
    if (tid!=0 || row>=(size_t)T) return;
    uint32_t dst=sa+slot*65536, bar=ma+slot*8;
    expect(bar);
    bulk(dst,resid+row*8192,32768,bar);
    bulk(dst+32768,attn+row*8192,16384,bar);
    bulk(dst+49152,ffn+(MAP?(size_t)prow[k]:row)*8192,16384,bar);
  };
  prefetch(blockIdx.x,0,0);
  if (MAP) {  // row scales and top-8 ids of this CTA's tokens (complete once the wait returned)
    for (int k=tid; k<nk; k+=NT) prps[k]=rps[prow[k]];
    for (int k=tid; k<8*nk; k+=NT) ids_out[(blockIdx.x+(size_t)(k>>3)*gridDim.x)*8+(k&7)]=ids_in[(size_t)prow[k>>3]*8+(k&7)];
    __syncthreads();
  }
  int iteration=0;
  for (size_t row=blockIdx.x; row<(size_t)T; row+=gridDim.x,++iteration) {
    int slot=iteration&1;
    wait(ma+slot*8,(iteration>>1)&1);
    const unsigned char* cur=stage+slot*65536;
    float x[VEC], b[VEC];
    #pragma unroll
    for (int g=0; g<GROUPS; ++g) {
      float4 v=reinterpret_cast<const float4*>(cur)[NT*g+tid];
      x[4*g]=v.x; x[4*g+1]=v.y; x[4*g+2]=v.z; x[4*g+3]=v.w;
      ld4(reinterpret_cast<const __nv_bfloat16*>(cur+32768),NT*g+tid,b+4*g);
    }
    prefetch(row+gridDim.x,slot^1,iteration+1);
    const float rpa=MAP?prps[iteration]:rps[row];
    #pragma unroll
    for (int j=0; j<VEC; ++j) {
      float nb=bfr(b[j]*rpa);
      x[j]=ar[j]*x[j]+br[j]*nb;
    }
    #pragma unroll
    for (int g=0; g<GROUPS; ++g)
      ld4(reinterpret_cast<const __nv_bfloat16*>(cur+49152),NT*g+tid,b+4*g);
    #pragma unroll
    for (int g=0; g<GROUPS; ++g) {
      float w[4]; reg4(post_r[2*g],post_r[2*g+1],w);
      #pragma unroll
      for (int q=0; q<4; ++q) b[4*g+q]=bfr(b[4*g+q]*w[q]);
    }
    float ss=0.f;
    #pragma unroll
    for (int j=0; j<VEC; ++j) ss+=b[j]*b[j];
    float rp=rsqrtf(bsum(ss,red[0])/8192.f+post_eps);
    float ss2=0.f;
    #pragma unroll
    for (int g=0; g<GROUPS; ++g) {
      #pragma unroll
      for (int q=0; q<4; ++q) {
        int j=4*g+q;
        float nb=bfr(b[j]*rp);
        x[j]=cr[j]*x[j]+dr[j]*nb;
        ss2+=x[j]*x[j];
      }
      reinterpret_cast<float4*>(nst+row*8192)[NT*g+tid]=
          make_float4(x[4*g],x[4*g+1],x[4*g+2],x[4*g+3]);
    }
    float rq=rsqrtf(bsum(ss2,red[1])/8192.f+pre_eps);
    #pragma unroll
    for (int g=0; g<GROUPS; ++g) {
      float w[4]; reg4(pre_r[2*g],pre_r[2*g+1],w);
      uint2 v;
      *reinterpret_cast<__nv_bfloat162*>(&v.x)=__floats2bfloat162_rn(x[4*g]*rq*w[0],x[4*g+1]*rq*w[1]);
      *reinterpret_cast<__nv_bfloat162*>(&v.y)=__floats2bfloat162_rn(x[4*g+2]*rq*w[2],x[4*g+3]*rq*w[3]);
      reinterpret_cast<uint2*>(out+row*8192)[NT*g+tid]=v;
    }
  }
}

// remat2_k: the same per-element math and block reductions as remat_k with a different memory schedule. The
// 160 KB of per-column parameters sit in shared memory: alpha / beta of both boundaries as float4 [2048] each, then
// the post / pre weights as uint2 [2048] each. The next token's three rows are loaded straight into registers before
// the current token's math (no TMA staging, no mbarriers); a copy-only version of this schedule moved the same bytes
// 5-6% faster than remat_k at 3012 and 16384 tokens.
template <bool MAP>
__global__ void __launch_bounds__(NT,1) remat2_k(const float* __restrict__ resid,
    const __nv_bfloat16* __restrict__ attn, const __nv_bfloat16* __restrict__ ffn,
    __nv_bfloat16* __restrict__ out, float* __restrict__ nst, const float* __restrict__ rps,
    const float* __restrict__ aa, const float* __restrict__ ba,
    const float* __restrict__ ab, const float* __restrict__ bb,
    const __nv_bfloat16* __restrict__ postw, const __nv_bfloat16* __restrict__ prew,
    float post_eps, float pre_eps, int T,
    const int* __restrict__ inv, const int* __restrict__ ids_in, int* __restrict__ ids_out) {
  __shared__ float red[2][WARPS];
  extern __shared__ __align__(16) unsigned char pbuf[];
  float4* sar=reinterpret_cast<float4*>(pbuf);
  float4* sbr=sar+2048; float4* scr=sar+4096; float4* sdr=sar+6144;
  uint2* spost=reinterpret_cast<uint2*>(sar+8192); uint2* spre=spost+2048;
  const int tid=threadIdx.x;
  #pragma unroll
  for (int g=0; g<GROUPS; ++g) {
    const int i=NT*g+tid;
    sar[i]=reinterpret_cast<const float4*>(aa)[i]; sbr[i]=reinterpret_cast<const float4*>(ba)[i];
    scr[i]=reinterpret_cast<const float4*>(ab)[i]; sdr[i]=reinterpret_cast<const float4*>(bb)[i];
    spost[i]=reinterpret_cast<const uint2*>(postw)[i]; spre[i]=reinterpret_cast<const uint2*>(prew)[i];
  }
  int row=blockIdx.x;
  int fr=row<T?(MAP?__ldg(inv+row):row):0;  // the map is complete before nr3_k ran: safe to read before the wait
  asm volatile("griddepcontrol.wait;" ::: "memory");
  float4 cs[GROUPS]; uint2 ca[GROUPS], cf[GROUPS];
  float crps=0.f;
  if (row<T) {
    #pragma unroll
    for (int g=0; g<GROUPS; ++g) {
      cs[g]=reinterpret_cast<const float4*>(resid+(size_t)row*8192)[NT*g+tid];
      ca[g]=reinterpret_cast<const uint2*>(attn+(size_t)row*8192)[NT*g+tid];
      cf[g]=reinterpret_cast<const uint2*>(ffn+(size_t)fr*8192)[NT*g+tid];
    }
    crps=rps[fr];
    if (MAP && tid<8) ids_out[(size_t)row*8+tid]=ids_in[(size_t)fr*8+tid];
  }
  __syncthreads();  // parameters in shared memory
  for (; row<T; row+=gridDim.x) {
    const int nrow=row+gridDim.x;
    float4 ns[GROUPS]; uint2 na[GROUPS], nf[GROUPS];
    float nrps=0.f;
    if (nrow<T) {
      const int nfr=MAP?__ldg(inv+nrow):nrow;
      #pragma unroll
      for (int g=0; g<GROUPS; ++g) {
        ns[g]=reinterpret_cast<const float4*>(resid+(size_t)nrow*8192)[NT*g+tid];
        na[g]=reinterpret_cast<const uint2*>(attn+(size_t)nrow*8192)[NT*g+tid];
        nf[g]=reinterpret_cast<const uint2*>(ffn+(size_t)nfr*8192)[NT*g+tid];
      }
      nrps=rps[nfr];
      if (MAP && tid<8) ids_out[(size_t)nrow*8+tid]=ids_in[(size_t)nfr*8+tid];
    }
    float x[VEC], b[VEC];
    #pragma unroll
    for (int g=0; g<GROUPS; ++g) {
      x[4*g]=cs[g].x; x[4*g+1]=cs[g].y; x[4*g+2]=cs[g].z; x[4*g+3]=cs[g].w;
      ld4(reinterpret_cast<const __nv_bfloat16*>(&ca[g]),0,b+4*g);
    }
    const float rpa=crps;
    #pragma unroll
    for (int g=0; g<GROUPS; ++g) {
      const float4 a4=sar[NT*g+tid], b4=sbr[NT*g+tid];
      const float av[4]={a4.x,a4.y,a4.z,a4.w}, bv[4]={b4.x,b4.y,b4.z,b4.w};
      #pragma unroll
      for (int q=0; q<4; ++q) {
        const int j=4*g+q;
        float nb=bfr(b[j]*rpa);
        x[j]=av[q]*x[j]+bv[q]*nb;
      }
    }
    #pragma unroll
    for (int g=0; g<GROUPS; ++g)
      ld4(reinterpret_cast<const __nv_bfloat16*>(&cf[g]),0,b+4*g);
    #pragma unroll
    for (int g=0; g<GROUPS; ++g) {
      const uint2 pw=spost[NT*g+tid];
      float w[4]; reg4(pw.x,pw.y,w);
      #pragma unroll
      for (int q=0; q<4; ++q) b[4*g+q]=bfr(b[4*g+q]*w[q]);
    }
    float ss=0.f;
    #pragma unroll
    for (int j=0; j<VEC; ++j) ss+=b[j]*b[j];
    float rp=rsqrtf(bsum(ss,red[0])/8192.f+post_eps);
    float ss2=0.f;
    #pragma unroll
    for (int g=0; g<GROUPS; ++g) {
      const float4 c4=scr[NT*g+tid], d4=sdr[NT*g+tid];
      const float cv[4]={c4.x,c4.y,c4.z,c4.w}, dv[4]={d4.x,d4.y,d4.z,d4.w};
      #pragma unroll
      for (int q=0; q<4; ++q) {
        int j=4*g+q;
        float nb=bfr(b[j]*rp);
        x[j]=cv[q]*x[j]+dv[q]*nb;
        ss2+=x[j]*x[j];
      }
      reinterpret_cast<float4*>(nst+(size_t)row*8192)[NT*g+tid]=
          make_float4(x[4*g],x[4*g+1],x[4*g+2],x[4*g+3]);
    }
    float rq=rsqrtf(bsum(ss2,red[1])/8192.f+pre_eps);
    #pragma unroll
    for (int g=0; g<GROUPS; ++g) {
      const uint2 pw=spre[NT*g+tid];
      float w[4]; reg4(pw.x,pw.y,w);
      uint2 v;
      *reinterpret_cast<__nv_bfloat162*>(&v.x)=__floats2bfloat162_rn(x[4*g]*rq*w[0],x[4*g+1]*rq*w[1]);
      *reinterpret_cast<__nv_bfloat162*>(&v.y)=__floats2bfloat162_rn(x[4*g+2]*rq*w[2],x[4*g+3]*rq*w[3]);
      reinterpret_cast<uint2*>(out+(size_t)row*8192)[NT*g+tid]=v;
    }
    #pragma unroll
    for (int g=0; g<GROUPS; ++g) { cs[g]=ns[g]; ca[g]=na[g]; cf[g]=nf[g]; }
    crps=nrps;
  }
}
} // namespace

void remat_init() {
  cudaFuncSetAttribute(remat_k<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,128*1024);
  cudaFuncSetAttribute(remat_k<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,128*1024);
  cudaFuncSetAttribute(remat2_k<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,160*1024);
  cudaFuncSetAttribute(remat2_k<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,160*1024);
}
void remat_run(torch::Tensor resid,torch::Tensor attn,torch::Tensor ffn,torch::Tensor out,
    torch::Tensor nst,torch::Tensor rps,torch::Tensor aa,torch::Tensor ba,torch::Tensor ab,torch::Tensor bb,
    torch::Tensor postw,torch::Tensor prew,double post_eps,double pre_eps,int64_t grid,
    torch::Tensor inv,torch::Tensor ids_in,torch::Tensor ids_out,int64_t v2) {
  int T=(int)resid.size(0);
  const bool map=inv.numel()>0;
  TORCH_CHECK(!map || (inv.numel()>=T && T<=(int64_t)RMAX*grid && ids_in.numel()>=8*T && ids_out.numel()>=8*T), "remat map");
  cudaLaunchAttribute attrs[1];
  attrs[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed=1;
  cudaLaunchConfig_t cfg={dim3((unsigned)grid),dim3(NT),(size_t)(v2?160:128)*1024,at::cuda::getCurrentCUDAStream(),attrs,1};
  cudaLaunchKernelEx(&cfg,v2?(map?remat2_k<true>:remat2_k<false>):(map?remat_k<true>:remat_k<false>),resid.data_ptr<float>(),(const __nv_bfloat16*)attn.data_ptr(),
      (const __nv_bfloat16*)ffn.data_ptr(),(__nv_bfloat16*)out.data_ptr(),nst.data_ptr<float>(),rps.data_ptr<float>(),
      aa.data_ptr<float>(),ba.data_ptr<float>(),ab.data_ptr<float>(),bb.data_ptr<float>(),
      (const __nv_bfloat16*)postw.data_ptr(),(const __nv_bfloat16*)prew.data_ptr(),(float)post_eps,(float)pre_eps,T,
      map?(const int*)inv.data_ptr<int>():nullptr,map?(const int*)ids_in.data_ptr<int>():nullptr,
      map?ids_out.data_ptr<int>():nullptr);
}
