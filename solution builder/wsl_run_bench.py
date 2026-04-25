import json, math, os, subprocess, time

BIN_PATH = './subm_2/mlsys'
BENCH_DIR = './benchmarks'
OUT_DIR = './output_wsl'

os.makedirs(OUT_DIR, exist_ok=True)

# Asegurar que sea ejecutable
if os.path.exists(BIN_PATH):
    os.chmod(BIN_PATH, 0o755)

TIMEOUTS = {
    'mlsys-2026-1' :2,  'mlsys-2026-2' :2,  'mlsys-2026-3' :2,  'mlsys-2026-4' :2,
    'mlsys-2026-5' :5,  'mlsys-2026-6' :5,  'mlsys-2026-7' :5,  'mlsys-2026-8' :5,
    'mlsys-2026-9' :15, 'mlsys-2026-10':15, 'mlsys-2026-11':15, 'mlsys-2026-12':15,
    'mlsys-2026-13':30, 'mlsys-2026-14':30, 'mlsys-2026-15':30, 'mlsys-2026-16':30,
    'mlsys-2026-17':60, 'mlsys-2026-18':60, 'mlsys-2026-19':60, 'mlsys-2026-20':60,
}

def get_timeout(bench_name):
    return TIMEOUTS.get(bench_name.replace('.json', ''), 60)

# ─ Modelo de latencia (roofline) ─────────────────────────────────────────────
def compute_latency(problem, solution):
    widths       = problem['widths']
    heights      = problem['heights']
    inputs_list  = problem['inputs']
    outputs_list = problem['outputs']
    base_costs   = problem['base_costs']
    op_types     = problem['op_types']
    slow_bw      = problem['slow_memory_bandwidth']
    native_w, native_h = problem['native_granularity']
    num_ops = len(base_costs)
    subgraphs         = solution['subgraphs']
    granularities     = solution['granularities']
    tensors_to_retain = solution['tensors_to_retain']
    traversal_orders  = solution['traversal_orders']

    tensor_producer = {}
    all_consumed = set()
    for op_id in range(num_ops):
        for t in outputs_list[op_id]: tensor_producer[t] = op_id
        for t in inputs_list[op_id]:  all_consumed.add(t)
    graph_outputs = set(range(len(widths))) - all_consumed

    computed_latencies = []
    for sg_idx, sg_ops in enumerate(subgraphs):
        w, h, k = granularities[sg_idx]
        trav = traversal_orders[sg_idx]
        prev_retained = set(tensors_to_retain[sg_idx - 1]) if sg_idx > 0 else set()
        curr_retained = set(tensors_to_retain[sg_idx])

        ephemeral = set()
        for op_id in sg_ops:
            for t_id in outputs_list[op_id]:
                if t_id not in graph_outputs and any(
                        t_id in inputs_list[op2] for op2 in sg_ops if op2 != op_id):
                    ephemeral.add(t_id)

        max_W = max((widths[t] for op in sg_ops for t in outputs_list[op]), default=0)
        max_H = max((heights[t] for op in sg_ops for t in outputs_list[op]), default=0)
        n_tiles_w = max(1, math.ceil(max_W / w)) if max_W else 1
        n_tiles_h = max(1, math.ceil(max_H / h)) if max_H else 1
        n_tiles   = n_tiles_w * n_tiles_h

        K = max((widths[inputs_list[op][0]] for op in sg_ops if op_types[op]=='MatMul'), default=0)
        n_k = max(1, math.ceil(K / k)) if K else 1

        pad = math.ceil(w / native_w) * math.ceil(h / native_h)
        mm_comp_step = sum(base_costs[op]*pad/n_k for op in sg_ops if op_types[op]=='MatMul')
        pw_comp      = sum(base_costs[op]*pad     for op in sg_ops if op_types[op]!='MatMul')

        boundary_inputs = {}
        for op_id in sg_ops:
            for i, t_id in enumerate(inputs_list[op_id]):
                if t_id in ephemeral or t_id in prev_retained or t_id in boundary_inputs:
                    continue
                if op_types[op_id] == 'MatMul':
                    boundary_inputs[t_id] = ('mm_lhs', widths[t_id]*h, 0) if i==0 else ('mm_rhs', w*k, w*k)
                else:
                    boundary_inputs[t_id] = ('pw_in', w*h, 0)

        out_size = sum(w*h for op in sg_ops for t in outputs_list[op]
                       if t not in ephemeral and t not in curr_retained)

        tile_seq  = trav if trav is not None else list(range(n_tiles))
        has_reuse = trav is not None
        total_lat = 0.0
        prev_row, prev_col = -1, -1

        for tile_pos, tile_idx in enumerate(tile_seq):
            row, col = tile_idx // n_tiles_w, tile_idx % n_tiles_w
            for kstep in range(n_k):
                compute = mm_comp_step + (pw_comp if kstep == n_k-1 else 0)
                mem_in  = 0.0
                for t_id, (role, first_load, mid_load) in boundary_inputs.items():
                    if kstep > 0:
                        mem_in += mid_load
                    elif role == 'mm_lhs':
                        if not (has_reuse and tile_pos > 0 and row == prev_row): mem_in += first_load
                    elif role == 'mm_rhs':
                        if not (has_reuse and tile_pos > 0 and col == prev_col): mem_in += first_load
                    else:
                        mem_in += first_load
                mem_out = out_size if kstep == n_k-1 else 0.0
                total_lat += max(compute, (mem_in + mem_out) / slow_bw)
            prev_row, prev_col = row, col

        computed_latencies.append(total_lat)
    return computed_latencies

# ─ Validación ─────────────────────────────────────────────────────────────────
def validate_solution(problem, solution):
    errors, warnings = [], []
    widths       = problem['widths']
    heights      = problem['heights']
    inputs_list  = problem['inputs']
    outputs_list = problem['outputs']
    base_costs   = problem['base_costs']
    op_types     = problem['op_types']
    fast_mem     = problem['fast_memory_capacity']
    num_ops      = len(base_costs)
    num_tensors  = len(widths)

    required = ['subgraphs','granularities','tensors_to_retain','traversal_orders','subgraph_latencies']
    for field in required:
        if field not in solution: errors.append(f'Missing field: {field}')
    if errors: return False, errors, warnings, 0

    sgs   = solution['subgraphs']
    grans = solution['granularities']
    retns = solution['tensors_to_retain']
    travs = solution['traversal_orders']
    lats  = solution['subgraph_latencies']
    n_sg  = len(sgs)

    for fname, lst in [('granularities',grans),('tensors_to_retain',retns),
                       ('traversal_orders',travs),('subgraph_latencies',lats)]:
        if len(lst) != n_sg: errors.append(f'{fname}: longitud {len(lst)} != {n_sg}')
    if errors: return False, errors, warnings, 0

    tensor_producer = {}
    all_produced, all_consumed = set(), set()
    for op_id in range(num_ops):
        for t in outputs_list[op_id]: tensor_producer[t] = op_id; all_produced.add(t)
        for t in inputs_list[op_id]:  all_consumed.add(t)
    graph_outputs = set(range(num_tensors)) - all_consumed

    covered = set()
    for sg_ops in sgs:
        for op_id in sg_ops:
            if not (0 <= op_id < num_ops): errors.append(f'Op inválido: {op_id}')
            covered.add(op_id)
    missing = set(range(num_ops)) - covered
    if missing: errors.append(f'Ops no cubiertos: {sorted(missing)}')

    for i, gran in enumerate(grans):
        if len(gran) != 3: errors.append(f'SG {i}: granularity debe tener 3 elementos')
        elif gran[0]<=0 or gran[1]<=0 or gran[2]<=0:
            errors.append(f'SG {i}: granularity no positiva: {gran}')

    for i, retain in enumerate(retns):
        accessible = set()
        for op_id in sgs[i]:
            for t in outputs_list[op_id]: accessible.add(t)
            for t in inputs_list[op_id]:  accessible.add(t)
        if i > 0: accessible |= set(retns[i-1])
        for t_id in retain:
            if not (0 <= t_id < num_tensors): errors.append(f'SG {i}: tensor inválido {t_id}')
            elif t_id not in accessible: errors.append(f'SG {i}: tensor {t_id} no accesible al retener')

    for sg_idx, sg_ops in enumerate(sgs):
        seen = set()
        for op_id in sg_ops:
            if op_id in seen: errors.append(f'SG {sg_idx}: op duplicado {op_id}')
            seen.add(op_id)
        op_pos = {op_id: pos for pos, op_id in enumerate(sg_ops)}
        op_set = set(sg_ops)
        for op_id in sg_ops:
            for t_id in inputs_list[op_id]:
                if t_id in tensor_producer and tensor_producer[t_id] in op_set:
                    prod = tensor_producer[t_id]
                    if op_pos[prod] >= op_pos[op_id]:
                        errors.append(f'SG {sg_idx}: orden topológico violado — op {op_id} necesita op {prod}')

    op_first_sg = {}
    for sg_idx, sg_ops in enumerate(sgs):
        for op_id in sg_ops:
            if op_id not in op_first_sg: op_first_sg[op_id] = sg_idx
    for sg_idx, sg_ops in enumerate(sgs):
        op_set = set(sg_ops)
        for op_id in sg_ops:
            for t_id in inputs_list[op_id]:
                if t_id not in tensor_producer: continue
                prod = tensor_producer[t_id]
                if prod in op_set: continue
                earliest = op_first_sg.get(prod)
                if earliest is None or earliest >= sg_idx:
                    errors.append(f'SG {sg_idx}: op {op_id} necesita salida de op {prod} no planificado antes')

    for sg_idx, sg_ops in enumerate(sgs):
        if sg_idx >= len(grans) or len(grans[sg_idx]) != 3: continue
        w, h, k = grans[sg_idx]
        ephemeral = set()
        for op_id in sg_ops:
            for t_id in outputs_list[op_id]:
                if t_id not in graph_outputs and any(
                        t_id in inputs_list[op2] for op2 in sg_ops if op2 != op_id):
                    ephemeral.add(t_id)
        prev_ret_set = set(retns[sg_idx-1]) if sg_idx > 0 else set()
        prev_ret_sz  = sum(widths[t]*heights[t] for t in prev_ret_set)
        mm_lhs, mm_rhs, pw_in, out_t = {}, {}, {}, {}
        for op_id in sg_ops:
            if op_types[op_id] == 'MatMul':
                lhs, rhs = inputs_list[op_id][0], inputs_list[op_id][1]
                if lhs not in ephemeral and lhs not in prev_ret_set: mm_lhs[lhs] = True
                if rhs not in ephemeral and rhs not in prev_ret_set: mm_rhs[rhs] = True
                for t in outputs_list[op_id]:
                    if t not in ephemeral: out_t[t] = True
            else:
                for t in inputs_list[op_id]:
                    if t not in ephemeral and t not in prev_ret_set: pw_in[t] = True
                for t in outputs_list[op_id]:
                    if t not in ephemeral: out_t[t] = True
        joint_ws = (sum(widths[t]*h for t in mm_lhs) + len(mm_rhs)*w*k +
                    len(pw_in)*w*h + len(out_t)*w*h)
        total_ws = joint_ws + prev_ret_sz
        ret_sz = sum(widths[t]*heights[t] for t in retns[sg_idx]) if sg_idx < len(retns) else 0
        if ret_sz > fast_mem:
            errors.append(f'SG {sg_idx}: tensores retenidos ({ret_sz}) > fast_mem ({fast_mem})')
        if total_ws > fast_mem:
            errors.append(f'SG {sg_idx}: OOM — WS={joint_ws}+prev_ret={prev_ret_sz}={total_ws} > {fast_mem} [w={w},h={h},k={k}]')

    if n_sg > 0:
        bad = set(retns[-1]) & graph_outputs
        if bad: errors.append(f'Salidas del grafo {sorted(bad)} retenidas al final (deben estar en slow memory)')

    for i, lat in enumerate(lats):
        if lat <= 0: errors.append(f'SG {i}: latencia debe ser positiva, es {lat}')

    try:
        comp_lats = compute_latency(problem, solution)
        for i, (rep, comp) in enumerate(zip(lats, comp_lats)):
            if comp > 0 and abs(rep - comp) / comp > 0.01:
                warnings.append(f'SG {i}: reportado={rep:.1f} roofline={comp:.1f} (err {abs(rep-comp)/comp:.1%})')
    except Exception as e:
        warnings.append(f'No se pudieron verificar latencias vs roofline: {e}')

    return len(errors)==0, errors, warnings, sum(lats)

# ─ Ejecutar benchmark ─────────────────────────────────────────────────────────
def run_benchmark(binary, bench_path, out_path, timeout_sec):
    cmd = ['wsl', binary, bench_path.replace('\\', '/'), out_path.replace('\\', '/')]
    try:
        t0 = time.time()
        r  = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec + 1)
        elapsed = time.time() - t0
        if r.returncode != 0:
            return False, elapsed, f'Exit {r.returncode}: {r.stderr.strip()[:200]}'
        return True, elapsed, None
    except subprocess.TimeoutExpired:
        return False, timeout_sec, 'TIMEOUT'
    except Exception as e:
        return False, 0, str(e)

# ─ Precondiciones ─────────────────────────────────────────────────────────────
assert os.path.exists(BIN_PATH),  'Binario no encontrado'
bench_files = sorted([f for f in os.listdir(BENCH_DIR) if f.endswith('.json')])
assert bench_files, 'Sin benchmarks'

import platform
W = 120
print('=' * W)
print(f'{"MLSys 2026 — Evaluador WSL":^{W}}')
print(f'{"Cores por defecto del binario | Timeouts oficiales":^{W}}')
print('=' * W)
print(f'  Sistema:    {platform.platform()}')
print(f'  Binario:    {BIN_PATH}  ({os.path.getsize(BIN_PATH):,} bytes)')
print(f'  Benchmarks: {bench_files}')

print(f'\n{"-" * W}')
print(f'  {"Benchmark":<22} {"Status":<9} {"Tiempo(s)":<10} {"TL(s)":<7} {"En TL?":<7}'
      f' {"Ciclos":>16} {"#SG":>5} {"Válido":>7} {"OOM":>5}  Notas')
print(f'  {"-"*22} {"-"*9} {"-"*10} {"-"*7} {"-"*7} {"-"*16} {"-"*5} {"-"*7} {"-"*5}  {"-"*35}')

total_cycles = 0.0
n_valid = 0
results = {}

for bench_file in bench_files:
    bench_name = bench_file.replace('.json', '')
    bench_path = os.path.join(BENCH_DIR, bench_file)
    out_path   = os.path.join(OUT_DIR, bench_file)
    timeout    = get_timeout(bench_name)

    print(f'  {bench_name:<22} ', end='', flush=True)
    success, elapsed, err = run_benchmark(BIN_PATH, bench_path, out_path, timeout)

    if not success:
        status = 'TIMEOUT' if err == 'TIMEOUT' else 'FAIL'
        print(f'{status:<9} {elapsed:<10.2f} {timeout:<7} {"NO":<7}'
              f' {"—":>16} {"—":>5} {"—":>7} {"—":>5}  {(err or "")[:35]}')
        results[bench_name] = {'status': status, 'elapsed': elapsed, 'within_tl': False, 'valid': False}
        continue

    try:
        with open(bench_path) as f: problem  = json.load(f)
        with open(out_path)   as f: solution = json.load(f)
        valid, errs, warns, rep_total = validate_solution(problem, solution)

        n_sg     = len(solution.get('subgraphs', []))
        in_tl    = elapsed <= timeout
        has_oom  = any('OOM' in e for e in errs)
        status_s = 'OK' if valid else 'INVALID'
        notes    = (errs[0][:35] if errs else (warns[0][:35] if warns else ''))

        if valid: total_cycles += rep_total; n_valid += 1

        print(f'{status_s:<9} {elapsed:<10.2f} {timeout:<7} {"SI" if in_tl else "NO":<7}'
              f' {rep_total:>16.1f} {n_sg:>5} {"SI" if valid else "NO":>7} {"SI" if has_oom else "no":>5}  {notes}')

        for e in errs[:3]:  print(f'    ! {e}')
        for ww in (warns[:2] if not errs else []): print(f'    ~ {ww}')

        results[bench_name] = {
            'status': status_s, 'elapsed': elapsed, 'timeout': timeout,
            'within_tl': in_tl, 'cycles': rep_total if valid else None,
            'n_subgraphs': n_sg, 'valid': valid, 'errors': errs, 'warnings': warns}

    except Exception as e:
        print(f'{"ERROR":<9} {elapsed:<10.2f} {timeout:<7} {"—":<7}'
              f' {"—":>16} {"—":>5} {"—":>7} {"—":>5}  {str(e)[:35]}')
        results[bench_name] = {'status': 'ERROR', 'elapsed': elapsed, 'valid': False}

# ─ Totales ────────────────────────────────────────────────────────────────────
all_within = all(r.get('within_tl', False) for r in results.values())
all_valid  = all(r.get('valid', False)      for r in results.values())
pass_s     = 'PASS' if (all_within and all_valid) else 'FAIL'
total_time = sum(r.get('elapsed', 0) for r in results.values())

print(f'  {"-"*22} {"-"*9} {"-"*10} {"-"*7} {"-"*7} {"-"*16} {"-"*5} {"-"*7} {"-"*5}  {"-"*35}')
print(f'  {"TOTAL":<22} {pass_s:<9} {total_time:<10.2f} {"":<7}'
      f' {"TL: "+("SI" if all_within else "NO"):<7}'
      f' {total_cycles:>16.1f} {"":>5}'
      f' {"Val: "+("SI" if all_valid else "NO"):>7} {"":>5}'
      f'  {n_valid}/{len(bench_files)} OK')

print(f'\n{"═" * W}')
print(f'{"RESULTADO FINAL":^{W}}')
print(f'{"═" * W}')
print(f'  Total ciclos : {total_cycles:>20,.1f}')
print(f'  Total tiempo : {total_time:>20.2f} s')
print(f'  Válidos      : {n_valid}/{len(bench_files)}')
print(f'  Competencia  : {pass_s}')
print('═' * W)
