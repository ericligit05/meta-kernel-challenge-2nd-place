# What did not work

A strong kernel engineer would expect each idea below to speed up this layer, yet each lost when we measured it in the
layer, or gained too little to keep. Most lost to contention between work that shares an SM, or to the power limit
described in section 3.1 of the [README](../README.md).

## Ideas that should have won

| idea | measured result | why it lost |
|---|---|---|
| Run the narrow expert tiles in a guest warpgroup beside the wide wgmma tiles | The hosts' MMA rate falls to 66 to 72%, and the best forms lose 6 to 9us at turn 1 | The guest's co-run cost exceeds the narrow-tile time it removes |
| A 5th ring stage in the turn 1 expert kernel | 0.83 to 0.86us faster | Smaller than the 1 to 2.5us by which a renamed extension alone can shift a short shape (README section 6), so we read it as no real gain from lookahead beyond 4 stages and kept 4 |
| A 5th ring stage in the 16K expert kernel | First read +95us slower | The stage index `gch % NS` became a real division at a depth that is not a power of two, and those ~10 integer instructions per chunk cost ~85us. With running counters a 6th stage is worth only ~23us |
| A second producer thread to spread the TMA issue at turn 1 | +1.4 to +5.3us in every form | The only spare warp is a copier, and removing one copier costs 3.4 to 3.6us |
| 2-CTA multicast of the token tiles in the 16K expert kernel | Saved 61 to 69us with the old loop order, about 10us once each token tile visits all of an expert's row blocks | Multicast cuts L2-to-SM bytes, and with the final loop order those bytes no longer bound the run time |
| Fuse byte-bound kernels into a GEMM | Turn 4 skeleton, 151.3us fused against 146.5us serial | The byte kernels also run at the 700W cap, so none of their traffic hides |
| Dense GEMM configurations that spend fewer joules | At 16K, a pooler configuration 24us faster standalone ran 43us slower in the layer | Cutting joules saves little time in the layer (see below) |
| Place each SM's data in its near half of the L2 | A local stream probe runs 5.1% faster, but the layer's kernels gained at most ~0.03% of score | The H200 copies a far line into the near half on first touch, so only first-touch bytes help |
| Stream-K for full attention | +8.7us at turn 4, +8.8us at turn 1 | Partial-tile epilogues and the last-arriver merge cost more than the balancing saves |

## Energy depends on the operand bits

Under the power limit, a tensor core's energy depends on the bits it multiplies, as we measured on the layer's own
GEMM operands at turn 4.

- Mantissa bits dominate. Zeroing the activations' mantissas cuts GEMM energy by 11 to 12%, zeroing their low 4 bits
  by 5 to 6%, and fixing every exponent by 2 to 3.5%.
- Zeros save energy only on the weight side. Zeroing half the weights saves 4.6 to 6%, and zeroing half the
  activations saves under 0.5%.
- The NVFP4 block scale fills in mantissa bits, and those bits cost about 36% of the expert phase's energy at the
  turn 4 sliding shape. Feeding the tensor cores unscaled E2M1 values is worth about 2.6% of score as an oracle, but
  no exact form exists. The scale belongs to one weight row and one 16-wide k block, so applying it outside the MMA
  costs one FP32 FFMA per 16 MACs, which saturates the FMA pipe.
- INT8 and FP8 deliver only 1.3x and 1.8x the operations per joule of BF16 under the cap, far below the 6x an exact
  multi-limb design needs.

In the layer, removing 1% of the energy saves only about 0.25% of the time, while a removed cycle saves 0.7 to 1.0
cycles. A change that spends cycles to save joules therefore loses, as every lower-energy GEMM configuration did.
