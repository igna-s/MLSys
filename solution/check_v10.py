import json, glob, os

build_dir = os.path.join(r'd:\GitHub\Golge algo\MLSys\solution', 'build')
files = sorted(glob.glob(os.path.join(build_dir, 'output_*.json')))
total = 0.0
for f in files:
    with open(f) as fp:
        data = json.load(fp)
    lats = data.get('subgraph_latencies', [])
    s = sum(lats)
    total += s
    name = os.path.basename(f).replace('output_', '')
    nsub = len(lats)
    print(f'{name:30s}  score={s:>15,.2f}  subgraphs={nsub}')

print('-' * 60)
print(f'{"TOTAL":30s}  score={total:>15,.2f}')
if total < 10_000_000:
    print('TARGET ACHIEVED! Under 10M cycles!')
else:
    print(f'NEED TO REDUCE BY {total - 10_000_000:,.0f} more cycles')
