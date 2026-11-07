"""Recalc chrF++ in per_sample.csv + metrics.json for all scenarios."""
import csv, glob, json, math, os, statistics, sys

sys.path.insert(0, 'FE_AGAL/metrics')
from chrf import chrf_plus

SCENARIOS = ['A_in_domain', 'B1_semi_no_dict', 'B2_semi_with_dict', 'C_out_of_domain']

# 1. Recalc per_sample.csv
for scenario in SCENARIOS:
    p = f'FE_AGAL/results/{scenario}/per_sample.csv'
    if not os.path.exists(p):
        print(f'[SKIP] {p} not found')
        continue
    rows = list(csv.DictReader(open(p, encoding='utf-8', newline='')))
    for r in rows:
        r['chrF++'] = str(chrf_plus(r['hyp'], r['ref']))
    fields = list(rows[0].keys())
    with open(p, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f'[CSV] {scenario}: recalc {len(rows)} rows')

# 2. Recalc metrics.json
for scenario in SCENARIOS:
    p = f'FE_AGAL/results/{scenario}/metrics.json'
    if not os.path.exists(p):
        continue
    rows = list(csv.DictReader(open(
        f'FE_AGAL/results/{scenario}/per_sample.csv', encoding='utf-8', newline='')))
    vals = [float(r['chrF++']) for r in rows]
    n = len(vals)
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(n) if n > 0 else 0.0
    d = json.load(open(p, encoding='utf-8'))
    old = d.get('chrF++', {}).get('mean', 'N/A')
    d['chrF++'] = {'mean': m, 'std': sd, 'ci95_low': m - ci, 'ci95_high': m + ci}
    json.dump(d, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'[JSON] {scenario}: chrF++ {old} -> {m:.4f}')

print('\nDone. Now run: python FE_AGAL/analyze.py')
