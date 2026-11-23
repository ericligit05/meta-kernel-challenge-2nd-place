"""Check that the animated figures move at the rates they claim, from the keyframes in the SVGs themselves.

    python3 figures/check_animation.py                       # all three figures
    python3 figures/check_animation.py path/to/architecture.svg

architecture.svg (one CTA, 64 equal k-steps): the accumulator's drawn width must grow linearly over the mainloop
(max deviation from a straight line under 2% of full width; no segment's rate off the mean by more than 10%), and
the "k n/64" counter labels must appear in order, 1 to 64, at evenly spaced times; and the staging write, the copier
store and a refill must each last their estimated real length in k-steps (REAL in architecture_animation.py, which
cites the code and profile it comes from), within 10%.
layer_pipeline.svg: every cursor that carries data-kx (px per data unit) must move at one constant rate in data
units per second, the same for all cursors.
speedup_breakdown.svg: reports each moving element's rates; that figure makes no claim about time.
Exits non-zero if any check fails.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def keyframes(svg):
    """class -> (period_s, [(t_s, props)]) with props holding o, tx, ty, sx as present."""
    periods = {m[0]: float(m[1]) for m in re.findall(r'\.(a\d+)\{animation:k\1 ([\d.]+)s', svg)}
    out = {}
    for name, body in re.findall(r'@keyframes k(a\d+)\{(.*?\})\}', svg):
        per = periods[name]
        frames = []
        for pct, decl in re.findall(r'([\d.]+)%\{([^}]*)\}', body):
            p = {}
            m = re.search(r'opacity:([\d.]+)', decl)
            if m:
                p['o'] = float(m[1])
            m = re.search(r'translate\(([-\d.]+)px,([-\d.]+)px\)', decl)
            if m:
                p['tx'], p['ty'] = float(m[1]), float(m[2])
            m = re.search(r'translateX\(([-\d.]+)px\)', decl)
            if m:
                p['tx'] = float(m[1])
            m = re.search(r'scaleX\(([-\d.]+)\)', decl)
            if m:
                p['sx'] = float(m[1])
            frames.append((float(pct) / 100 * per, p))
        out[name] = (per, frames)
    return out


def value(frames, key, t):
    """Linear interpolation between keyframes (the animations use timing function linear)."""
    pts = [(ft, p[key]) for ft, p in frames if key in p]
    for (a, va), (b, vb) in zip(pts, pts[1:]):
        if a <= t <= b:
            return va if b == a else va + (vb - va) * (t - a) / (b - a)
    return pts[-1][1]


def elements(svg):
    """class -> the element's opening tag."""
    return {m[1]: m[0] for m in re.finditer(r'<[a-z]+ [^>]*class="(?:fb )?(a\d+)"[^>]*>', svg)}


def on_intervals(per, frames):
    """Durations of the stretches with opacity >= 0.5, merging one that wraps around the end of the loop."""
    pts = [(t, p['o']) for t, p in frames if 'o' in p]
    out, start = [], None
    for t, o in pts:
        if o >= 0.5 and start is None:
            start = t
        elif o < 0.5 and start is not None:
            out.append([start, t])
            start = None
    if start is not None:
        if out and out[0][0] == 0:
            out[0][0] = start - per
        else:
            out.append([start, per])
    return [b - a for a, b in out]


def check_architecture(svg):
    ok = True
    kf, els = keyframes(svg), elements(svg)
    acc = next(c for c, tag in els.items() if 'data-ev="acc"' in tag)
    base_w = float(re.search(r'width="([\d.]+)"', els[acc])[1])
    per, fr = kf[acc]
    sx = [(t, p['sx']) for t, p in fr if 'sx' in p]
    full = max(v for _, v in sx)
    # the mainloop: from the last keyframe at zero width before the peak to the first keyframe at full width
    i1 = next(i for i, (_, v) in enumerate(sx) if v >= full - 1e-9)
    i0 = max(i for i, (_, v) in enumerate(sx[:i1]) if v <= 1e-9)
    t0, t1 = sx[i0][0], sx[i1][0]
    n = 384  # a multiple of the 32 windows
    ts = [t0 + (t1 - t0) * k / n for k in range(n + 1)]
    ws = [base_w * value(fr, 'sx', t) for t in ts]
    W = base_w * full
    dev = max(abs(w - W * (t - t0) / (t1 - t0)) for t, w in zip(ts, ws)) / W
    mean = W / (t1 - t0)
    # rate over 32 equal windows, and over every keyframe segment longer than 1% of the mainloop
    win = [(ws[k + n // 32] - ws[k]) / (ts[k + n // 32] - ts[k]) for k in range(0, n, n // 32)]
    seg = [(vb - va) * base_w / (b - a) for (a, va), (b, vb) in zip(sx[i0:i1], sx[i0 + 1:i1 + 1]) if b - a > 0.01 * (t1 - t0)]
    worst = max(abs(r / mean - 1) for r in win + seg)
    print(f'  accumulator: {len(ts)} samples over {t0:.3f}s..{t1:.3f}s of a {per:.3f}s loop; full width {W:.1f}px, '
          f'mean rate {mean:.2f}px/s')
    print(f'  max deviation from a straight line {100 * dev:.2f}% of full width (limit 2%)')
    print(f'  worst segment rate vs mean {100 * worst:.1f}% (limit 10%); window rates {min(win):.2f}..{max(win):.2f}px/s')
    if dev >= 0.02 or worst > 0.10:
        print('  FAIL: accumulator fill is not linear')
        ok = False
    # k counter: label text and the time it turns on
    labs = []
    for c, tag in els.items():
        if 'data-ev="k label"' not in tag:
            continue
        text = re.search(r'class="%s"[^>]*>.*?<text[^>]*>([^<]*)</text>' % c, svg)[1]
        on = next(t for t, p in kf[c][1] if p.get('o', 0) >= 0.999)
        labs.append((on, text))
    labs.sort()
    nums = [re.fullmatch(r'k (\d+)/64', s) for _, s in labs]
    bad = [s for (_, s), m in zip(labs, nums) if not m]
    seq = [int(m[1]) for m in nums if m]
    gaps = [b[0] - a[0] for a, b in zip(labs, labs[1:])]
    spread = (max(gaps) - min(gaps)) / (sum(gaps) / len(gaps)) if gaps else 0
    print(f'  k labels: {len(labs)}, first {labs[0][1]!r}, last {labs[-1][1]!r}; step {min(gaps):.3f}..{max(gaps):.3f}s')
    if bad or seq != list(range(1, 65)) or spread > 0.05:
        print(f'  FAIL: k labels not 1..64 in order at even steps (non-numeric {bad}, spread {100 * spread:.0f}%)')
        ok = False
    # the counter must advance at the accumulator's rate: one label step is one 64th of the fill time
    if gaps:
        step = sum(gaps) / len(gaps)
        print(f'  k label step {step:.4f}s vs fill time / 64 = {(t1 - t0) / 64:.4f}s')
        if abs(step / ((t1 - t0) / 64) - 1) > 0.02:
            print('  FAIL: the counter and the accumulator advance at different rates')
            ok = False
        # durations against their estimated real length, on the k-step's scale
        sys.path.insert(0, HERE)
        from architecture_animation import REAL
        for ev, name in (('staging write', 'staging write'), ('store', 'copier store'), ('tma s0', 'refill')):
            c = next(c for c, tag in els.items() if f'data-ev="{ev}"' in tag)
            ds = on_intervals(*kf[c])
            d = sum(ds) / len(ds)
            ratio = d / step / REAL[name]
            print(f'  {name}: on screen {1000 * d:.0f}ms = {d / step:.2f} k-steps; real estimate {REAL[name]:.2f} '
                  f'k-steps ({100 * (ratio - 1):+.0f}%)')
            if abs(ratio - 1) > 0.10:
                print(f'  FAIL: {name} is not on the k-step time scale')
                ok = False
    return ok


def moving_rates(kf, cls, key):
    per, fr = kf[cls]
    pts = [(t, p[key]) for t, p in fr if key in p]
    return [((vb - va) / (b - a), a, b, va, vb) for (a, va), (b, vb) in zip(pts, pts[1:]) if b > a and abs(vb - va) > 0.5]


def check_layer_pipeline(svg):
    ok = True
    kf, els = keyframes(svg), elements(svg)
    rates = []
    for c, tag in els.items():
        m = re.search(r'data-kx="([\d.]+)"', tag)
        if not m:
            continue
        kx = float(m[1])
        segs = [r for r in moving_rates(kf, c, 'tx') if r[0] > 0]  # forward motion; the jump back is hidden at opacity 0
        r = [s[0] / kx for s in segs]
        ev = re.search(r'data-ev="([^"]*)"', tag)[1]
        print(f'  {c} ({ev}): {len(segs)} moving segments, '
              + ', '.join(f'{x:.2f}us/s' for x in r))
        rates += r
    mean = sum(rates) / len(rates)
    worst = max(abs(x / mean - 1) for x in rates)
    print(f'  all cursors: mean {mean:.2f}us/s, worst deviation {100 * worst:.2f}% (limit 1%)')
    if worst > 0.01:
        print('  FAIL: cursors do not share one constant rate')
        ok = False
    return ok


def report_speedup(svg):
    kf = keyframes(svg)
    for c in sorted(kf, key=lambda s: int(s[1:])):
        for key in ('tx', 'ty', 'sx'):
            for r, a, b, va, vb in moving_rates(kf, c, key):
                print(f'  {c} {key}: {va:.1f} -> {vb:.1f} over {a:.3f}..{b:.3f}s = {r:.1f}/s')
    return True


CHECKS = {'architecture.svg': check_architecture, 'layer_pipeline.svg': check_layer_pipeline,
          'speedup_breakdown.svg': report_speedup}


def main():
    paths = sys.argv[1:] or [os.path.join(HERE, n) for n in CHECKS]
    ok = True
    for p in paths:
        name = next((n for n in CHECKS if os.path.basename(p).endswith(n)), 'architecture.svg')
        print(f'{p} ({name} check)')
        ok &= CHECKS[name](open(p).read())
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
