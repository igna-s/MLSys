import json

for name in ['mlsys-2026-1','mlsys-2026-5','mlsys-2026-9','mlsys-2026-13','mlsys-2026-17']:
    d = json.load(open(f'benchmarks/{name}.json'))
    w, h = d['widths'], d['heights']
    prod, cons = set(), set()
    for o in d['outputs']:
        for t in o: prod.add(t)
    for i in d['inputs']:
        for t in i: cons.add(t)
    gi = cons - prod
    go = prod - cons
    ib = sum(w[t]*h[t] for t in gi)
    ob = sum(w[t]*h[t] for t in go)
    bw = d['slow_memory_bandwidth']
    tc = sum(d['base_costs'])
    mm = (ib + ob) / bw
    print(f"{name}: ops={len(d['inputs'])} gi={len(gi)} go={len(go)} in={ib:>12,} out={ob:>12,} min_mem={mm:>12,.0f} compute={tc:>8,} theoretical_min={max(tc,mm):>12,.0f}")
