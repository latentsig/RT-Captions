#!/usr/bin/env python3
"""Diarization error rate, stdlib only.  Run as: python3 -I der.py ref.rttm hyp.rttm [options]

DER = (missed + false alarm + confusion) / total reference speech, with a collar around every
reference segment boundary (default 0.25 s, each side), overlap scored normally (a frame with 2
reference speakers counts 2 in the denominator), and the optimal ONE-TO-ONE mapping of hypothesis
labels to reference speakers (maximum co-occurrence time, exhaustive DP over reference speakers).

Options:
  --collar S        collar in seconds (default 0.25)
  --start T --end T score only this part of the file (default: from the first to the last reference
                    segment; speech outside it is unannotated, so it is not scored)
  --names REF=Name,...  identity report: how much reference speech is labelled with the right
                    person name / a wrong name / "Voice N" or "..." (unnamed) / not labelled
  --self-test       check the maths on a tiny hand-made pair of RTTMs
"""
import sys
from collections import defaultdict


def read_rttm(path):
    segs = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 8 and p[0] == "SPEAKER":
                segs.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
    return segs


def is_unnamed(label):
    return label.startswith("Voice_") or label.startswith("Voice ") or label in ("…", "...", "?")


def sweep(ref, hyp, collar, start, end):
    """Yields (duration, ref speakers, hyp speakers) for every elementary interval outside the collars."""
    cuts = {start, end}
    for a, b, _ in ref + hyp:
        cuts.update((a, b))
    bounds = [x for a, b, _ in ref for x in (a, b)]
    for x in bounds:
        cuts.update((x - collar, x + collar))
    cuts = sorted(c for c in cuts if start <= c <= end)
    collars = sorted((x - collar, x + collar) for x in set(bounds))
    ev = defaultdict(list)  # time -> [(kind, label, +1/-1)]
    for a, b, s in ref:
        ev[a].append(("r", s, 1)); ev[b].append(("r", s, -1))
    for a, b, s in hyp:
        ev[a].append(("h", s, 1)); ev[b].append(("h", s, -1))
    act = {"r": defaultdict(int), "h": defaultdict(int)}
    times = sorted(set(ev) | set(cuts))
    ci = 0
    covered = 0  # number of collars covering the current interval (sweep over sorted collar starts/ends)
    cev = sorted([(a, 1) for a, _ in collars] + [(b, -1) for _, b in collars])
    k = 0
    for t0, t1 in zip(times, times[1:]):
        for kind, s, d in ev.get(t0, ()):
            act[kind][s] += d
        while k < len(cev) and cev[k][0] <= t0 + 1e-9:
            covered += cev[k][1]; k += 1
        if t1 <= start or t0 >= end or covered > 0 or t1 - t0 <= 1e-9:
            continue
        yield (min(t1, end) - max(t0, start),
               {s for s, n in act["r"].items() if n > 0}, {s for s, n in act["h"].items() if n > 0})


def best_mapping(co, refs, hyps):
    """hyp -> ref, maximising total co-occurrence, one-to-one (DP over subsets of reference speakers)."""
    n = len(refs)
    best = {0: (0.0, ())}  # mask of used refs -> (score, ((hyp, ref idx), ...))
    for h in hyps:
        nxt = dict(best)
        for mask, (sc, pairs) in best.items():
            for i, r in enumerate(refs):
                if mask >> i & 1:
                    continue
                v = sc + co[r].get(h, 0.0)
                m2 = mask | 1 << i
                if m2 not in nxt or v > nxt[m2][0] + 1e-12:
                    nxt[m2] = (v, pairs + ((h, r),))
        best = nxt
    return dict(max(best.values(), key=lambda x: x[0])[1])


def score(ref, hyp, collar=0.25, start=None, end=None, names=None):
    start = start if start is not None else min(a for a, _, _ in ref)
    end = end if end is not None else max(b for _, b, _ in ref)
    ref = [s for s in ref if s[1] > start and s[0] < end]
    hyp = [s for s in hyp if s[1] > start and s[0] < end]
    refs = sorted({s for _, _, s in ref})
    hyps = sorted({s for _, _, s in hyp})
    if len(refs) > 8:
        raise SystemExit("more than 8 reference speakers")
    co = {r: defaultdict(float) for r in refs}
    tot = {"ref": 0.0, "miss": 0.0, "fa": 0.0, "conf": 0.0}
    ivs = list(sweep(ref, hyp, collar, start, end))
    for d, R, H in ivs:
        for r in R:
            for h in H:
                co[r][h] += d
    mp = best_mapping(co, refs, hyps)  # hyp -> ref
    per = {r: defaultdict(float) for r in refs}
    hyp_time = defaultdict(float)
    ident = defaultdict(float)
    for d, R, H in ivs:
        nr, nh = len(R), len(H)
        mapped = {mp[h] for h in H if h in mp}
        correct = len(mapped & R)
        tot["ref"] += d * nr
        tot["miss"] += d * max(0, nr - nh)
        tot["fa"] += d * max(0, nh - nr)
        tot["conf"] += d * (min(nr, nh) - correct)
        for h in H:
            hyp_time[h] += d
        for r in R:
            p = per[r]
            p["ref"] += d
            if r in mapped:
                p["correct"] += d
            elif H:
                p["conf"] += d
            else:
                p["miss"] += d
            if names:
                named = [h for h in H if not is_unnamed(h)]
                if not H:
                    ident["none"] += d
                elif names.get(r) in H:
                    ident["right"] += d
                elif named:
                    ident["wrong"] += d
                else:
                    ident["unnamed"] += d
    r = tot["ref"] or 1.0
    der = (tot["miss"] + tot["fa"] + tot["conf"]) / r
    # purity: how much of each hypothesis label is one reference speaker; coverage: the reverse
    purity = sum(max((co[x].get(h, 0.0) for x in refs), default=0.0) for h in hyps)
    purity /= sum(sum(co[x].get(h, 0.0) for x in refs) for h in hyps) or 1.0
    cov = sum(max(co[x].values(), default=0.0) for x in refs) / sum(sum(co[x].values()) for x in refs) if any(co[x] for x in refs) else 0.0
    return {"der": der, "tot": tot, "mapping": mp, "per": per, "hyp_time": hyp_time, "purity": purity,
            "coverage": cov, "ident": ident, "refs": refs, "hyps": hyps}


def report(res, names=None):
    t = res["tot"]
    r = t["ref"] or 1.0
    print(f"DER {100 * res['der']:.1f}%  (missed {100 * t['miss'] / r:.1f}%, false alarm {100 * t['fa'] / r:.1f}%, "
          f"confusion {100 * t['conf'] / r:.1f}%)  reference speech {t['ref']:.0f} s")
    big = [h for h, s in res["hyp_time"].items() if s >= 1.0]
    print(f"purity {100 * res['purity']:.1f}%  coverage {100 * res['coverage']:.1f}%  "
          f"hypothesis labels {len(res['hyps'])} ({len(big)} with >= 1 s of speech) vs {len(res['refs'])} reference speakers")
    inv = {r: h for h, r in res["mapping"].items()}
    print(f"{'ref speaker':12} {'speech s':>8} {'correct%':>9} {'confused%':>9} {'missed%':>8}  mapped label")
    for s in res["refs"]:
        p = res["per"][s]
        x = p["ref"] or 1.0
        print(f"{s:12} {p['ref']:8.1f} {100 * p['correct'] / x:9.1f} {100 * p['conf'] / x:9.1f} {100 * p['miss'] / x:8.1f}  {inv.get(s, '-')}")
    extra = [(h, s) for h, s in sorted(res["hyp_time"].items(), key=lambda x: -x[1]) if h not in res["mapping"]]
    if extra:
        print("unmapped labels: " + ", ".join(f"{h} {s:.0f} s" for h, s in extra))
    if names:
        i = res["ident"]
        tot = sum(i.values()) or 1.0
        print(f"identity: right name {100 * i['right'] / tot:.1f}%  wrong name {100 * i['wrong'] / tot:.1f}%  "
              f"unnamed {100 * i['unnamed'] / tot:.1f}%  not labelled {100 * i['none'] / tot:.1f}%  (of reference speaker-time)")


def self_test():
    # reference: A 0-10, B 10-20 (hand made).  hypothesis: x 0-9 (A), y 9-20 (B) with the boundary 1 s early.
    ref = [(0.0, 10.0, "A"), (10.0, 20.0, "B")]
    hyp = [(0.0, 9.0, "x"), (9.0, 20.0, "y")]
    r = score(ref, hyp, collar=0.0)
    assert abs(r["tot"]["conf"] - 1.0) < 1e-6 and abs(r["tot"]["miss"]) < 1e-6 and abs(r["tot"]["fa"]) < 1e-6, r["tot"]
    assert abs(r["der"] - 0.05) < 1e-6, r["der"]
    assert r["mapping"] == {"x": "A", "y": "B"}
    # the 0.25 s collar forgives 0.25 s each side of the boundary at 10: confusion in 9-10 is 0.75 s outside it
    r = score(ref, hyp, collar=0.25)
    assert abs(r["tot"]["conf"] - 0.75) < 1e-6, r["tot"]
    # missed + false alarm: hyp only covers 0-5 of A, and talks 25-30
    r = score(ref, [(0.0, 5.0, "x"), (25.0, 30.0, "x")], collar=0.0, end=30.0)
    assert abs(r["tot"]["miss"] - 15.0) < 1e-6 and abs(r["tot"]["fa"] - 5.0) < 1e-6, r["tot"]
    # overlap counts twice in the reference: A and B both speak 0-4, hypothesis has one speaker
    r = score([(0.0, 4.0, "A"), (0.0, 4.0, "B")], [(0.0, 4.0, "x")], collar=0.0)
    assert abs(r["der"] - 0.5) < 1e-6, r["der"]
    # one-to-one mapping: two hypothesis labels for one speaker cannot both map to it
    r = score([(0.0, 10.0, "A")], [(0.0, 6.0, "x"), (6.0, 10.0, "y")], collar=0.0)
    assert abs(r["der"] - 0.4) < 1e-6 and r["purity"] == 1.0, (r["der"], r["purity"])
    # identity report
    r = score(ref, [(0.0, 10.0, "Ann"), (10.0, 15.0, "Bob"), (15.0, 20.0, "Voice_3")], collar=0.0, names={"A": "Ann", "B": "Ben"})
    assert abs(r["ident"]["right"] - 10.0) < 1e-6 and abs(r["ident"]["wrong"] - 5.0) < 1e-6 and abs(r["ident"]["unnamed"] - 5.0) < 1e-6, dict(r["ident"])
    print("der.py self-test ok")


def main(argv):
    if "--self-test" in argv:
        return self_test()
    opts = {}
    pos = []
    it = iter(argv)
    for a in it:
        if a.startswith("--"):
            opts[a] = next(it)
        else:
            pos.append(a)
    if len(pos) != 2:
        raise SystemExit(__doc__)
    names = dict(kv.split("=") for kv in opts["--names"].split(",")) if "--names" in opts else None
    res = score(read_rttm(pos[0]), read_rttm(pos[1]), float(opts.get("--collar", 0.25)), float(opts["--start"]) if "--start" in opts else None,
                float(opts["--end"]) if "--end" in opts else None, names)
    report(res, names)


if __name__ == "__main__":
    main(sys.argv[1:])
