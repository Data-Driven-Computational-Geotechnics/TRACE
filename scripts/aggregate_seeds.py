#!/usr/bin/env python3
"""解析 score_all_2d 打印的表(多个日志), 按变体聚合多种子 -> 均值±标准差 + 原始每种子数据。
用法: python tools/aggregate_seeds.py <log1> [<log2> ...]
变体名 = 去掉 _sN 后缀 (st_s0/st_s1 -> st)。"""
import sys, re
import numpy as np

# 解析行: label kind params roll final deposit
LINE = re.compile(r'^(\S+)\s+(trace|gns|tgnn)\s+([\d,]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$')
rows = {}
for lg in sys.argv[1:]:
    try:
        for ln in open(lg):
            m = LINE.match(ln.strip())
            if m:
                label, kind, params, roll, final, dep = m.groups()
                rows[label] = dict(kind=kind, params=int(params.replace(",", "")),
                                   roll=float(roll), final=float(final), dep=float(dep))
    except FileNotFoundError:
        print(f"[warn] 缺日志 {lg}")

def variant(label):
    return re.sub(r'_s\d+$', '', label)

groups = {}
for label, r in rows.items():
    groups.setdefault(variant(label), []).append((label, r))

def agg(vals):
    a = np.array(vals)
    return a.mean(), a.std(ddof=0)   # 总体标准差(种子少, ddof=0)

def fmt(vals):
    m, s = agg(vals)
    return f"{m:.3f}±{s:.3f}"

def report(title, order, namemap):
    print(f"\n{'='*76}\n{title}\n{'='*76}")
    print(f"{'模型':14s} {'seeds':6s} {'roll_avg (mean±std)':22s} {'final':20s} {'deposit':20s} {'params':>10s}")
    for v in order:
        if v not in groups:
            print(f"{namemap.get(v,v):14s} {'—':6s} (未训完/无数据)"); continue
        items = sorted(groups[v], key=lambda x: x[0])
        rolls = [r['roll'] for _, r in items]; finals = [r['final'] for _, r in items]; deps = [r['dep'] for _, r in items]
        p = items[0][1]['params']; n = len(items)
        print(f"{namemap.get(v,v):14s} {n:<6d} {fmt(rolls):22s} {fmt(finals):20s} {fmt(deps):20s} {p:>10,}")
    # 原始每种子
    print(f"\n原始每种子数据:")
    for v in order:
        if v not in groups: continue
        for label, r in sorted(groups[v], key=lambda x: x[0]):
            print(f"  {label:16s} roll={r['roll']:.4f}  final={r['final']:.4f}  deposit={r['dep']:.4f}")

report("表1  基线对比 (2D, 30轨迹×300步, 帧1种子长时自回归)",
       ["st", "gns", "nmgns"], {"st": "Trace(本文)", "gns": "GNS", "nmgns": "NMGNS"})
report("表2  记忆消融 (2D, 同协议)",
       ["none", "spatial", "gru", "st"],
       {"none": "无记忆", "spatial": "纯空间", "gru": "纯时间(GRU)", "st": "时空(本文)"})
