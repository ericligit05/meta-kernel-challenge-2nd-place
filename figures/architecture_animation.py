"""Rewrite the animated CTA panel of figures/architecture.svg (one tile of the 16K expert kernel, moe_fused<256, 1>).

    python3 figures/architecture_animation.py            # rewrites the panel and its keyframes in place
    python3 figures/architecture_animation.py --table    # prints each element's lit intervals instead

What it draws, from src/kernels/moe_expert_gemm.cu (moe_fused, TN = 256, MB = 1):
- a CTA tile is 128 weight rows of one expert x 256 tokens, over K = 4096 in 64 chunks of 64 (k-steps);
- a stage of the 4-stage ring holds one chunk: 256 x 64 BF16 tokens (32KB) and 8 x 576B of NVFP4 weight units
  (4.5KB), so the token tile is drawn about 7x the weight tile;
- the producer keeps the ring full, TMA refilling a stage as soon as the consumers release it, across tiles;
- per chunk, the consumers decode one k16 step at a time and issue its wgmma right away (MOE_IL = 1), so the tensor
  pipe starts after the first quarter of A is decoded;
- wg_wait<1> after each commit: chunk c's wgmma is issued while chunk c-1's still runs, and the stage of chunk c-1 is
  released only then; the last stage is released after wg_wait<0>, before the epilogue;
- the consumers write the accumulator to the staging buffer, and the 3 copier warps store it while the next tile's
  first chunk runs. The CTA drawn is one of the down projection (moe_fused_run's phase 2 launch), whose copiers send
  each token row to its own route's slot (one bulk copy per row), so the store scatters the tile across the output;
  8 drawn rows stand for the tile's 256, each moving at one speed along its path.
The accumulator is the CTA's output tile, 128 weight rows x 256 tokens in FP32.
The whole loop runs on one time scale, P = 0.11s of screen per k-step, which is about 1.2us on the H200 (REAL below),
so the epilogue, the copier store and a refill take their estimated real length in k-steps. Every one of the 64
k-steps takes P, so the accumulator grows linearly from the first wgmma to the last. Within a k-step the order of
read, decode and issue is the code's; their split of the step is illustrative.
figures/check_animation.py checks the fill, the k counter and each duration against REAL.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SVG = os.path.join(HERE, 'architecture.svg')

# ---- real durations, estimated from the code and the profile, in k-steps (1 k-step ~ 1.18us) ----
# k-step: moe_fused<256, 1> takes 3350 to 3755us per 16K layer (2 calls; figures/data/phase_profile.json, mean 3527us).
#   A GPU holds 32 experts and ~16384 routes (16384 tokens x top 8 / EP8), ~512 per expert = 2 token tiles of 256;
#   each expert has 96 row blocks of 128 (GEMM1 8192 gate/up rows, GEMM2 4096 rows; out1 / out2 strides 4096,
#   moe_expert_gemm.cu:470-491), so ~6144 tiles over 132 persistent CTAs: 46.5 tiles x 64 k-steps per CTA in 3527us.
# staging write: 128 x 256 BF16 = 64KB through stmatrix (moe_expert_gemm.cu:585-600, stsm4t :86) at 128B per clock of
#   shared memory = 512 clocks, 0.33us at the 1.57GHz the 16K shape holds (README.md:88): ~0.3 k-step.
# copier store: 256 bulk copies of one 256B token row (moe_expert_gemm.cu:484-491), 64KB at ~40GB/s per SM (an SM's
#   share of L2 / HBM write bandwidth): ~1.6us, ~1.3 k-steps. It overlaps the next tile's first k-steps.
# refill: one stage is a 32KB token box plus 8 x 576B weight units (moe_expert_gemm.cu:432-436): ~0.9us of L2 latency
#   and transfer, ~0.75 k-step.
REAL = {'k-step': 1.0, 'staging write': 0.3, 'copier store': 1.3, 'refill': 0.75}

# ---- schedule, in kernel time tau (s of screen) ----
P = 0.11        # chunk to chunk, the same for all 64 chunks (one k-step of the tensor pipe)
DEC0 = 0.015    # read of the stage into registers, before decode starts
DEC = 0.06      # decode of a chunk's four k16 steps (A grows 1/4 -> 1)
WG0 = DEC0 + DEC / 4   # chunk c's wgmma is issued once its first k16 step is decoded
LAG = 0.04      # chunk c is issued while chunk c-1 still runs; c runs on the tensor pipe from WG0 + LAG for P
WGD = LAG + P   # one chunk's wgmma, issue to completion; consecutive chunks overlap by LAG
REL = DEC0 + DEC       # chunk c-1's stage is released once chunk c is issued and c-1's wgmma is done
NCH = 64        # chunks per tile
T0 = 0.02       # chunk 1 starts here in the loop (before the end of the staging write, so the tile ends inside it)
STG = REAL['staging write'] * P    # consumers write the staging buffer (33ms: a flash, kept at its real length)
STORE = REAL['copier store'] * P   # copiers store it
TMA = REAL['refill'] * P           # a refill: TMA in flight until the tiles land
EPS = 0.004
FULL = 56 / 3.42  # accumulator scaleX when full (base width 3.42px of 56px)
# store scatter: 8 rows leave the staging box's left edge, run left, and drop into scattered route slots in HBM out
NROW = 8
ROW_YC = [358.5 + 5 * i for i in range(NROW)]               # centre line of row i, stacked down the staging box
SLOT_X = [560 + 36 * j for j in range(NROW)]                # slot j in HBM out, 14px wide, all left of the staging box
DEST = [5, 2, 7, 0, 4, 1, 6, 3]                             # row i lands in slot DEST[i]
ROW_W, ROW_H, SLOT_W, SLOT_Y = 10, 3, 14, 459
STG_X = 832                                                 # staging box's left edge


def s_tau(c):   # start of chunk c, kernel time
    return T0 + (c - 1) * P


def draw(tau):  # kernel time -> drawn time: one to one, nothing is compressed
    return tau


E_TAU = s_tau(NCH) + WG0 + WGD          # last wgmma done (wg_wait<0>)
E = draw(E_TAU)
PERIOD = E + STG - T0                   # next tile's chunk 1 starts when the staging write ends
assert WGD > P, 'consecutive wgmmas must overlap'
assert E < PERIOD, 'the mainloop ends inside the loop period'
assert REL >= WG0 + WGD - P, 'a stage is released after its chunk wgmma ends'
assert REL + TMA <= 3 * P, 'a refill lands before the stage is read again, 4 chunks later'
assert DEC0 + DEC < P, 'a chunk is decoded within its k-step'


def stage(c):
    return (c - 1) % 4


def release_tau(c):
    return E_TAU if c == NCH else s_tau(c + 1) + REL


# ---- elements: each is a list of (t0, t1, kind, args) intervals in drawn time ----
class El:
    def __init__(self, base):
        self.base = dict(base)  # o, tx, ty, sx
        self.ev = []            # (t0, t1, f) with f(u) -> dict of props for u in [0, 1]

    def add(self, t0, t1, f, knots=()):
        self.ev.append((t0, t1, f))
        self.knots = getattr(self, 'knots', set()) | {t0 + k * (t1 - t0) for k in knots}

    def flash(self, t0, t1, o=1.0):
        self.add(t0, t1, lambda u: {'o': o})

    def state(self, t):
        st = dict(self.base)
        for t0, t1, f in self.ev:
            for k in (-1, 0, 1):
                a, b = t0 + k * PERIOD, t1 + k * PERIOD
                if a <= t < b:
                    st.update(f((t - a) / (b - a)))
        return st

    def breaks(self):
        out = set()
        for t0, t1, f in self.ev:
            for t in (t0, t1):
                out |= {t % PERIOD, (t - EPS) % PERIOD}
        return out | {t % PERIOD for t in getattr(self, 'knots', ())}


els = {}
TX = {i: 627 + 92 * i for i in range(4)}
for i in range(4):
    n = 6 * i
    els[f'tma{i}'] = (n + 1, El({'o': 0}))
    els[f'w{i}'] = (n + 2, El({'o': 1, 'ty': 0}))
    els[f'x{i}'] = (n + 3, El({'o': 1, 'ty': 0}))
    els[f'empty{i}'] = (n + 4, El({'o': 0}))
    els[f'B{i}'] = (n + 5, El({'o': 0}))
    els[f'read{i}'] = (n + 6, El({'o': 0, 'tx': 0, 'ty': 0}))
els['4bit'] = (25, El({'o': 1}))
els['A'] = (26, El({'o': 1, 'sx': 1}))
els['Alabel'] = (27, El({'o': 1}))
els['wgmma0'] = (28, El({'o': 0}))
els['consumers'] = (29, El({'o': 0}))
els['acc'] = (30, El({'o': 1, 'sx': 0}))
els['stwrite'] = (31, El({'o': 0, 'ty': 0}))
els['stfull'] = (32, El({'o': 0}))
els['store'] = (33, El({'o': 0, 'tx': 0, 'ty': 0}))  # row 0 of the scatter
els['copiers'] = (34, El({'o': 0}))
els['producer'] = (35, El({'o': 0}))
els['wgmma1'] = (36, El({'o': 0}))
KLAB = [(c, f'k {c}/64') for c in range(1, NCH + 1)]
for j, (c, _) in enumerate(KLAB):
    els[f'k{j}'] = (37 + j, El({'o': 0}))
for i in range(1, NROW):
    els[f'row{i}'] = (37 + NCH + i - 1, El({'o': 0, 'tx': 0, 'ty': 0}))
for j in range(NROW):
    els[f'land{j}'] = (37 + NCH + NROW - 1 + j, El({'o': 0}))  # outline flash on slot j as its row lands
E_ = {k: v[1] for k, v in els.items()}
lit = []  # (element, start, end) for the table

# the consumers issue continuously through the mainloop and then write the staging buffer
E_['consumers'].flash(draw(s_tau(1)), E + STG)
for c in range(1, NCH + 1):
    i = stage(c)
    t = draw(s_tau(c))
    # consumers read the stage: the weight piece drops into the 4-bit registers, the stage's B and arrows
    E_[f'read{i}'].add(t, t + DEC0, lambda u, i=i: {'o': 1, 'tx': (-1 - 92 * i) * min(1, max(0, (u - 0.35) / 0.45)),
                                                     'ty': 65 * min(1, u / 0.35) + 23 * max(0, (u - 0.8) / 0.2)}, knots=(0.35, 0.8))
    rel = draw(release_tau(c))
    E_[f'B{i}'].flash(t, rel)
    lit.append((f'B s{i} (chunk {c})', t, rel))
    E_['4bit'].add(t, t + DEC0, lambda u: {'o': 0})
    E_['A'].add(t + DEC0, t + DEC0 + DEC, lambda u: {'sx': 0.25 + 0.75 * u})
    E_['A'].add(t, t + DEC0, lambda u: {'o': 0, 'sx': 0.25})
    w0, w1 = draw(s_tau(c) + WG0), draw(s_tau(c) + WG0 + WGD)
    E_[f'wgmma{c % 2}'].flash(w0, w1)
    lit.append((f'wgmma (chunk {c})', w0, w1))
    lit.append((f'decode (chunk {c})', t + DEC0, t + DEC0 + DEC))
# counter: one label per chunk, each shown for exactly one k-step
for j, (c, _) in enumerate(KLAB):
    E_[f'k{j}'].flash(draw(s_tau(c)), draw(s_tau(c + 1)) if c != NCH else E)
# accumulator: the tensor pipe runs chunk c for P from s_tau(c) + WG0 + LAG, back to back, so the fill is one
# straight ramp from the first wgmma's start to the last one's end
a0 = draw(s_tau(1) + WG0 + LAG)
assert abs(E - (a0 + NCH * P)) < 1e-9
E_['acc'].add(a0, E, lambda u: {'sx': FULL * u})
E_['acc'].add(E, PERIOD + T0, lambda u: {'sx': 0})
# releases: empty flash, producer issues TMA, tiles land
for c in range(1, NCH + 1):
    i = stage(c)
    r = draw(release_tau(c))
    E_[f'empty{i}'].flash(r, r + 0.03)
    E_['producer'].flash(r, r + 0.05)
    E_[f'tma{i}'].flash(r, r + TMA)
    for k in ('w', 'x'):
        E_[f'{k}{i}'].add(r, r + 0.03, lambda u: {'o': 0, 'ty': -62})
        E_[f'{k}{i}'].add(r + 0.03, r + TMA, lambda u: {'o': 1, 'ty': -55.8 * (1 - u)})
    lit.append((f'release + refill s{i} (after chunk {c})', r, r + TMA))
# epilogue: staging write by the consumers, then the copiers store while the next tile starts
E_['consumers'].flash(E, E + STG)
E_['stwrite'].add(E, E + STG, lambda u: {'o': 1, 'ty': 104 * u})
E_['stfull'].flash(E + STG, E + STG + STORE)
E_['copiers'].flash(E + STG, E + STG + STORE)
for i in range(NROW):
    j = DEST[i]
    # an L path at one speed: left along the row's line, then down into its slot; the corner is a keyframe
    dx = SLOT_X[j] + (SLOT_W - ROW_W) / 2 - (STG_X - ROW_W)
    dy = SLOT_Y + (18 - ROW_H) / 2 - (ROW_YC[i] - ROW_H / 2)
    f = abs(dx) / (abs(dx) + abs(dy))
    E_['store' if i == 0 else f'row{i}'].add(
        E + STG, E + STG + STORE,
        lambda u, dx=dx, dy=dy, f=f: {'o': 1, 'tx': dx * min(1, u / f), 'ty': dy * max(0, (u - f) / (1 - f))}, knots=(f,))
    # HBM keeps what earlier tiles wrote, so the slot stays filled; the landing flashes its outline
    E_[f'land{j}'].flash(E + STG + STORE, E + STG + STORE + 0.05)
lit += [('staging write (consumers)', E, E + STG), ('copier store', E + STG, E + STG + STORE),
        ('next tile chunk 1', E + STG, E + STG + 0.1)]


def css():
    out = []
    for name, (n, el) in sorted(els.items(), key=lambda kv: kv[1][0]):
        ts = sorted(el.breaks() | {0.0})
        frames = []
        for t in ts + [PERIOD]:
            st = el.state(t % PERIOD if t < PERIOD else 0.0)
            tr = []  # only the properties the element animates, to keep the file small
            if 'tx' in el.base or 'ty' in el.base:
                tr.append(f"translate({st.get('tx', 0):.2f}px,{st.get('ty', 0):.2f}px)")
            if 'sx' in st:
                tr.append(f"scaleX({st['sx']:.4f})")
            tr = f"transform:{' '.join(tr)};" if tr else ''
            frames.append(f"{100 * t / PERIOD:.3f}%{{{tr}opacity:{st['o']:.3f}}}")
        cls = f'a{n}'
        out.append(f"@keyframes k{cls}{{{''.join(frames)}}}.{cls}{{animation:k{cls} {PERIOD:.3f}s linear 0.000s infinite}}")
    return ''.join(out)


INK, INK2, MUTED, BLUE, AMBER = '#2a2723', '#5e584f', '#8e877b', '#2f6fae', '#c78d46'


def static_markup():
    o = []
    for i in range(4):
        o.append(f'<rect x="{586 + 92 * i}" y="118" width="82" height="54" rx="3" fill="#fbf9f4" stroke="{MUTED}"/>')
    o.append('<rect x="598" y="252" width="12" height="38" fill="none" stroke="#9dbedd"/>'
             f'<rect x="640" y="252" width="48" height="38" fill="none" stroke="{AMBER}" stroke-width="1.6"/>'
             f'<path d="M613,271.0 L637,271.0" fill="none" stroke="{AMBER}" stroke-width="1.6" marker-end="url(#aa)"/>'
             f'<text x="614.0" y="243.0" font-size="14" fill="{AMBER}" text-anchor="start" style="font-weight:600">decode</text>'
             f'<text x="604.0" y="307.0" font-size="13" fill="{INK2}" text-anchor="middle">4-bit</text>'
             f'<text x="664.0" y="307.0" font-size="13" fill="{INK2}" text-anchor="middle">BF16</text>'
             f'<g data-inside="1"><rect x="724" y="252" width="76" height="38" rx="3" fill="#fbf9f4" stroke="{INK2}"/>'
             f'<text x="762.0" y="276.0" font-size="15" fill="{INK}" text-anchor="middle">wgmma</text></g>'
             f'<path d="M691,271.0 L721,271.0" fill="none" stroke="{INK2}" stroke-width="1.2" marker-end="url(#ai)"/>'
             f'<path d="M803,271.0 L829,271.0" fill="none" stroke="{INK2}" stroke-width="1.2" marker-end="url(#ai)"/>'
             f'<rect x="832" y="252" width="56" height="38" fill="none" stroke="{BLUE}"/>'
             f'<text x="868.0" y="243.0" font-size="13" fill="{INK2}" text-anchor="middle">accumulator, 128 × 256</text>'
             f'<path d="M860.0,293 L860.0,353" fill="none" stroke="{INK2}" stroke-width="1.2" marker-end="url(#ai)"/>'
             f'<rect x="832" y="356" width="56" height="38" fill="none" stroke="{BLUE}"/>'
             f'<text x="896.0" y="380.0" font-size="13" fill="{INK2}" text-anchor="start">staging</text>'
             + ''.join(f'<path d="M{STG_X},{ROW_YC[i]} H{SLOT_X[DEST[i]] + SLOT_W / 2:.1f} V{SLOT_Y - 1}" fill="none" '
                       f'stroke="#9dbedd" stroke-width="1" stroke-dasharray="3 3"/>' for i in range(NROW))
             + ''.join(f'<rect x="{x}" y="{SLOT_Y}" width="{SLOT_W}" height="18" fill="{BLUE}"/>' for x in SLOT_X)
             + f'<text x="{SLOT_X[-1] + SLOT_W + 8}.0" y="473.0" font-size="13.5" fill="{INK}" text-anchor="start">output</text>')
    return ''.join(o)


def anim_markup():
    """Animated elements; their static attributes are the frame shown when motion is reduced (chunk 1 on s0)."""
    o = []
    for i in range(4):
        x, n = 586 + 92 * i, 6 * i
        wx = x + 16  # weight tile 6px wide, token tile 44px: about 7:1, as 4.5KB to 32KB
        o.append(f'<path class="a{n + 1}" opacity="0" data-ev="tma s{i}" d="M{TX[i]:.1f},63 L{TX[i]:.1f},116" fill="none" stroke="{INK2}" stroke-width="1.3" marker-end="url(#ai)"/>')
        o.append(f'<rect class="a{n + 2}" opacity="1" data-ev="tile w s{i}" x="{wx}" y="126" width="6" height="38" fill="url(#hz)"/>')
        o.append(f'<rect class="a{n + 3}" opacity="1" data-ev="tile x s{i}" x="{x + 30}" y="126" width="44" height="38" fill="#cdc8bd"/>')
        o.append(f'<g class="a{n + 4}" opacity="0" data-ev="empty s{i}"><text x="{x + 52}.0" y="150.0" font-size="13.5" fill="{MUTED}" text-anchor="middle" style="font-style:italic">empty</text></g>')
        o.append(f'<g class="a{n + 5}" opacity="{1 if i == 0 else 0}" data-ev="B s{i}"><rect x="{x + 30}" y="126" width="44" height="38" fill="none" stroke="{BLUE}" stroke-width="1.6"/>'
                 f'<text x="{x + 52}.0" y="151.0" font-size="17" fill="{INK}" text-anchor="middle" style="font-weight:600">B</text>'
                 f'<path d="M{x + 52}.0,166 L762.0,250" fill="none" stroke="{BLUE}" stroke-width="1.2" stroke-dasharray="4 3" marker-end="url(#ab)"/>'
                 f'<path d="M{wx + 3}.0,166 V222 H604.0 V249" fill="none" stroke="{AMBER}" stroke-width="1.3" stroke-dasharray="4 3" marker-end="url(#aa)"/></g>')
        o.append(f'<rect class="a{n + 6}" opacity="0" data-ev="read s{i}" x="{wx}.0" y="150" width="6" height="14" fill="url(#hz)"/>')
    o.append('<rect class="a25" opacity="1" data-ev="4bit" x="598" y="252" width="12" height="38" fill="url(#hz)"/>')
    o.append(f'<rect class="fb a26" data-ev="A" x="640" y="252" width="48" height="38" fill="{AMBER}" fill-opacity="0.35"/>')
    o.append(f'<g class="a27" opacity="1" data-ev="A label"><text x="664.0" y="277.0" font-size="17" fill="{INK}" text-anchor="middle" style="font-weight:600">A</text></g>')
    o.append(f'<rect class="a28" opacity="1" data-ev="wgmma even" x="721" y="249" width="82" height="44" rx="5" fill="none" stroke="{INK}" stroke-width="1.5"/>')
    o.append(f'<rect class="a29" opacity="0" data-ev="consumers" x="447" y="223" width="102" height="42" rx="3" fill="none" stroke="{INK}" stroke-width="1.5"/>')
    o.append(f'<rect class="fb a30" data-ev="acc" x="832" y="252" width="3.42" height="38" fill="{BLUE}"/>')
    o.append(f'<rect class="a31" opacity="0" data-ev="staging write" x="832" y="252" width="56" height="38" fill="{BLUE}"/>')
    o.append(f'<rect class="a32" opacity="0" data-ev="staging full" x="832" y="356" width="56" height="38" fill="{BLUE}"/>')
    o.append(f'<rect class="a33" opacity="0" data-ev="store" x="{STG_X - ROW_W}" y="{ROW_YC[0] - ROW_H / 2}" width="{ROW_W}" height="{ROW_H}" fill="{BLUE}"/>')
    o.append(f'<rect class="a34" opacity="0" data-ev="copiers" x="447" y="339" width="78" height="24" rx="3" fill="none" stroke="{INK}" stroke-width="1.5"/>')
    o.append(f'<rect class="a35" opacity="0" data-ev="producer issues" x="447" y="109" width="30" height="24" rx="3" fill="none" stroke="{INK}" stroke-width="1.5"/>')
    o.append(f'<rect class="a36" opacity="0" data-ev="wgmma odd" x="718" y="246" width="88" height="50" rx="6" fill="none" stroke="{INK}" stroke-width="1.5"/>')
    for j, (c, lab) in enumerate(KLAB):
        o.append(f'<g class="a{37 + j}" opacity="{1 if j == 0 else 0}" data-ev="k label"><text x="896.0" y="276.0" font-size="13" fill="{INK2}" text-anchor="start">{lab}</text></g>')
    for i in range(1, NROW):
        o.append(f'<rect class="a{els[f"row{i}"][0]}" opacity="0" data-ev="store row {i}" x="{STG_X - ROW_W}" y="{ROW_YC[i] - ROW_H / 2}" width="{ROW_W}" height="{ROW_H}" fill="{BLUE}"/>')
    for j in range(NROW):
        o.append(f'<rect class="a{els[f"land{j}"][0]}" opacity="0" data-ev="slot {j} write" x="{SLOT_X[j] - 3}" y="{SLOT_Y - 3}" width="{SLOT_W + 6}" height="24" rx="3" fill="none" stroke="{INK}" stroke-width="1.5"/>')
    return ''.join(o)


def table():
    rows = sorted(lit, key=lambda r: r[1])
    print(f'period {PERIOD:.3f}s, k-step {P * 1000:.0f}ms, mainloop {NCH * P:.2f}s; chunk starts: '
          + ', '.join(f'{c}:{draw(s_tau(c)):.3f}' for c in list(range(1, 6)) + list(range(60, 65))))
    for name, a, b in rows:
        print(f'{name:40s} {a:7.3f} {b:7.3f}')


def main():
    if '--table' in sys.argv:
        table()
        return
    s = open(SVG).read()
    i = s.index('<rect x="586" y="118"')
    s = s[:i] + static_markup() + anim_markup() + '</svg>'
    fonts = ''.join(re.findall(r"@font-face\{[^}]*\}", s))
    base = ("text{font-family:'EBG',serif;font-variant-numeric:lining-nums proportional-nums;font-kerning:normal}"
            ".fb{transform-box:fill-box;transform-origin:left center}"
            "@media (prefers-reduced-motion:reduce){*{animation:none!important}}")
    s = re.sub(r'<style>.*?</style>', lambda m: '<style>' + fonts + base + css() + '</style>', s, count=1, flags=re.S)
    s = re.sub(r'(<text x="444.0" y="94.0"[^>]*>)[^<]*(</text>)',
               r'\g<1>one CTA, 16K expert kernel, down projection\g<2>', s, count=1)
    open(SVG, 'w').write(s)


if __name__ == '__main__':
    main()
