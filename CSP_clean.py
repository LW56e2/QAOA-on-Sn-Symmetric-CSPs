import pennylane as qml
import datetime
from pennylane import numpy as np
from itertools import combinations
import math
import time, os, json
import torch
torch.set_default_dtype(torch.float64)

def _pick_pl_device(n, device_kind=None, dev_kwargs=None):
    if dev_kwargs is None:
        dev_kwargs = {}
    if device_kind is not None:
        return qml.device(device_kind, wires=n, shots=None, **dev_kwargs)
    # Auto-pick with safe fallback
    if torch.cuda.is_available():
        try:
            return qml.device("lightning.gpu", wires=n, shots=None, **dev_kwargs)
        except Exception as e:
            print(f"[warn] lightning.gpu unavailable, falling back to CPU: {e}")
    return qml.device("lightning.qubit", wires=n, shots=None, **dev_kwargs)


class SymmetricCSP:
    """
    Build S_n-symmetric ell-N-m SATs with a planted solution s,
    """

    def __init__(self, n, spec, planted=None, include_permutations=False):
        """
        n: number of original variables
        spec: list of (ell, m) or (ell, m, weight) tuples, e.g. [(1,1), (3,1)] for 1N1 ∧ 3N1
        planted: optional tuple/list of length n with 0/1; default is all-zeros (wlog)
        include_permutations: if True multiply by m!(ell-m)! when m not in {0,ell} (Zhou-Montanaro does this)
        """
        self.n = int(n)
        self.include_permutations = bool(include_permutations)

        # normalize spec to triples (ell, m, weight)
        norm = []
        for item in spec:
            if len(item) == 2:
                ell, m = item; w = 1
            elif len(item) == 3:
                ell, m, w = item
            else:
                raise ValueError("spec items must be (ell, m) or (ell, m, weight)")
            ell = int(ell); m = int(m); w = int(w)
            if not (0 <= m <= ell <= self.n):
                raise ValueError(f"Bad (ell,m)=({ell},{m}); need 0 ≤ m ≤ ell ≤ n")
            if w <= 0:
                continue
            norm.append((ell, m, w))
        self.spec = norm

        if planted is None:
            self.s = tuple(0 for _ in range(n))
        else:
            assert len(planted) == n
            self.s = tuple(int(b) for b in planted)

        # prev version: built CNF clauses explicitly (enumeratively); deprecated now
        self.cnf = None

    @staticmethod
    def _comb(n, r):
        if r < 0 or r > n:
            return 0
        return math.comb(n, r)

    def _family_multiplier(self, ell, m):
        # Paper’s convention: include all permutations when m != 0, ell
        if self.include_permutations and (m not in (0, ell)):
            return math.factorial(m) * math.factorial(ell - m)
        return 1

    def analytic_c_of_k(self, k):
        """
        Violated-clause count at Hamming distance k, analytic
        c_{ell,m}(k) = [m!(ell-m)! if include_permutations and m∉{0,ell} else 1] * C(k,m)*C(n-k, ell-m)
        Weighted by 'weight'
        """
        n = self.n
        k = int(k)
        if not (0 <= k <= n):
            raise ValueError("k must be in [0, n]")
        total = 0
        for (ell, m, w) in self.spec:
            mult = self._family_multiplier(ell, m)
            total += w * mult * self._comb(k, m) * self._comb(n - k, ell - m)
        return total

    def analytic_ck(self):
        """
        Returns ks and c_k where ks = [0...n], and c_k[k] = analytic_c_of_k(k).
        """
        ks = list(range(self.n + 1))
        c_k = [self.analytic_c_of_k(k) for k in ks]
        return ks, c_k

    def cost_vs_hamming_weight_analytic(self):
        """
        analytic c(k)
        Returns: ks (0..n), c_exact (analytic), and a dummy agg dict to match prev versions
        """
        ks, c_k = self.analytic_ck()
        agg = {k: {"count": math.comb(self.n, k), "sum": c_k[k] * math.comb(self.n, k), "first": c_k[k]} for k in ks}
        return ks, c_k, agg

    def plot_cost_vs_hamming_weight_analytic(self, figsize=(6, 4)):
        """
        Matplotlib plot of c(k) vs k using the analytic method
        """
        import matplotlib.pyplot as plt
        ks, c_k = self.analytic_ck()
        plt.figure(figsize=figsize)
        plt.plot(ks, c_k, marker="o")
        norm = "with perms" if self.include_permutations else "no perms"
        plt.xlabel("Hamming weight k = |x ⊕ s|")
        plt.ylabel("cost c(k)")
        plt.title(f"Symmetric CSP {[(e,m) for (e,m,_) in self.spec]} at n={self.n} (analytic, {norm})")
        plt.grid(True)
        plt.show()

    # QUANTUM STUFF
    @staticmethod
    def _kraw(n, p, k):
        """Compute the Krawtchouk polynomial K_p(k;n)"""
        s = 0
        for r in range(0, p+1):
            s += ((-1)**r) * math.comb(k, r) * math.comb(n-k, p-r)
        return s

    def _fit_kraw_coeffs(self, max_degree=None):
        """
        Solve analytic c(k) = sum_{p=0..d} a_p K_p(k;n) for a_p.
        """
        ks, ck = self.analytic_ck()  # fast, exact integers
        n = self.n
        if max_degree is None:
            # Only need degree up to the largest ell in the spec
            d = max(ell for (ell, m, _) in self.spec) if self.spec else 0
        else:
            d = int(max_degree)
        # Build (n+1) x (d+1) matrix M[k,p] = K_p(k;n)
        M = np.zeros((n+1, d+1), dtype=float)
        for k in range(n+1):
            for p in range(d+1):
                M[k, p] = self._kraw(n, p, k)
        y = np.array(ck, dtype=float)
        a, *_ = np.linalg.lstsq(M, y, rcond=None)

        a = np.where(np.abs(a) < 1e-12, 0.0, a) # Clean tiny numerical noise
        return a  # a[0..d]

    # FIXED Hamiltonian builder (Krawtchouk expansion)
    def build_cost_hamiltonian_kraw(self, max_degree=None):
        n = self.n
        a = self._fit_kraw_coeffs(max_degree=max_degree)

        coeffs, ops = [], []

        def add(c, op_list):
            if abs(c) < 1e-12:
                return
            coeffs.append(float(c))
            if not op_list:
                ops.append(qml.Identity(0))
            else:
                op = op_list[0]
                for o in op_list[1:]:
                    op = op @ o
                ops.append(op)

        # p = 0, identity
        add(a[0], [])

        # p >= 1: include the planted string phase (-1)^{sum s_i over the subset}
        s = self.s
        for p in range(1, len(a)):
            ap = a[p]
            if abs(ap) < 1e-12:
                continue
            for idxs in combinations(range(n), p):
                phase = -1 if (sum(s[i] for i in idxs) % 2) else 1
                add(ap * phase, [qml.PauliZ(i) for i in idxs])

        return qml.Hamiltonian(coeffs, ops)

def decode_bits_pl(idx, n):
    # LITTLE-endian (matches PennyLane)
    return tuple((idx >> j) & 1 for j in range(n))

def bits_for_projector_from_index(idx, n):
    """
    Return the bitstring in WIRE ORDER expected by qml.Projector.
    Converts our PennyLane-probs little-endian indexing (LSB = wire 0)
    into big-endian (wire 0 first).
    """
    return tuple(reversed(decode_bits_pl(idx, n)))
def analytic_layer1_gamma_beta(n, ell=3, a=1): # heuristic params from Montanaro's paper
    """gamma1 = 2^(ell-2)/(ell-2a) * pi / n^(ell-1), beta1 = -pi/4"""
    g1 = (2**(ell-2) / (ell - 2*a)) * (np.pi / (n**(ell-1)))
    b1 = -np.pi/4
    return float(g1), float(b1)

def ground_shells_from_csp(csp):
    """Return the list of k with the lowest analytic cost (c(k))."""
    ks, ck = csp.analytic_ck()
    m = min(ck)
    return [k for k, v in zip(ks, ck) if v == m]

def build_success_projector_h(n, csp):
    """
    P_success = sum_{x in success} |x><x|
    Returned as a pennylane hamiltonian.
    """
    shells = set(ground_shells_from_csp(csp))
    coeffs, ops = [], []
    s = tuple(csp.s)
    for idx in range(1 << n):
        bits_le = decode_bits_pl(idx, n)  # little-endian for probs indexing
        k = sum(b ^ si for b, si in zip(bits_le, s))
        if k in shells:
            bits_be = bits_for_projector_from_index(idx, n)  # <-- key change
            ops.append(qml.Projector(bits_be, wires=range(n)))
            coeffs.append(1.0)

    if not ops:
        return qml.Hamiltonian([0.0], [qml.Identity(0)])

    return qml.Hamiltonian(coeffs, ops)



# experiments

def make_qaoa_success_scalar_fn_torch(n, cost_h, mixer_h, p, dev, proj_h):
    """qnode that returns the exact success probability <P_success> as a Torch scalar tensor."""
    @qml.qnode(dev, interface="torch", diff_method="best")
    def circuit(params):
        gammas = params[:p]; betas = params[p:]

        # |+>^n
        for w in range(n):
            qml.Hadamard(wires=w)

        # qaoa evolution (minus signs for Pennylane convention)
        for layer in range(p):
            qml.qaoa.cost_layer(-gammas[layer], cost_h)
            qml.qaoa.mixer_layer(-betas[layer],  mixer_h)

        # Single exact scalar: expectation of the projector Hamiltonian
        return qml.expval(proj_h)

    return circuit

def get_standard_mixer_h(n):
    """B = sum_i X_i as a pennylane Hamiltonian."""
    return qml.Hamiltonian([1.0]*n, [qml.PauliX(i) for i in range(n)])

def get_cost_and_mixer(csp):
    cost_h  = csp.build_cost_hamiltonian_kraw()
    mixer_h = get_standard_mixer_h(csp.n)

    return cost_h, mixer_h

@torch.no_grad()
def _project_betas_(params, p):
    """Clamp betas (last p entries) to [-pi, pi] in-place."""
    params[p:].clamp_(-np.pi, np.pi)

def train_qaoa_layered_torch(
    csp,
    p=2,
    mode="independent",     # "independent", "fix-first", and exclusive "fix-first-two" for p>=3
    steps=600,
    lr=5e-2,
    seed=1,
    device_kind=None,
    dev_kwargs=None,
    verbose=True,
):
    """
    PyTorch trainer for p-layer QAOA.
    Modes:
      - independent: train all (γ_t, β_t)
      - fix-first: freeze (γ_1, β_1)
      - fix-first-two: freeze (γ_1,β_1, γ_2,β_2) (for p>=3)
    """
    if dev_kwargs is None:
        dev_kwargs = {}
    n = csp.n

    # Build Hamiltonians
    cost_h, mixer_h = get_cost_and_mixer(csp)

    # PL device (statevector)
    dev = _pick_pl_device(n, device_kind=device_kind,
                          dev_kwargs={**(dev_kwargs or {}), "batch_obs": True})


    proj_h  = build_success_projector_h(n, csp)
    circuit = make_qaoa_success_scalar_fn_torch(n, cost_h, mixer_h, p, dev, proj_h)


    # Choose the dominant clause by largest ell
    ell_dom, m_dom, _ = max(csp.spec, key=lambda t: t[0])  # (ℓ, m, weight or implicit weight=1)
    a_dom = m_dom

    # Prefer a non-balanced term if possible
    if ell_dom == 2 * a_dom:
        non_balanced = [t for t in csp.spec if t[0] != 2 * t[1]]
        if non_balanced:
            # Use the largest-ell non-balanced term
            ell_dom, a_dom, _ = max(non_balanced, key=lambda t: t[0])
            balanced_dominant = False
        else:
            # Entire spec is balanced
            balanced_dominant = True
    else:
        balanced_dominant = False

    # Compute the p=1 optimal params (γ1, β1)
    if not balanced_dominant:
        # Standard Zhou-Montanaro seed for this (ell,a)
        g1, b1 = analytic_layer1_gamma_beta(n, ell=ell_dom, a=a_dom)
    else:
        # Fully balanced spec: your requested fallback
        # Try a=1; if that makes ell=2a again (only when ℓ==2), fall back to a=0
        a_fallback = 1
        if ell_dom == 2 * a_fallback:
            a_fallback = 0
        # This is just a seed
        # it doesn't need to match the spec's m
        # avoids division by zero
        g1, b1 = analytic_layer1_gamma_beta(n, ell=ell_dom, a=a_fallback)
        if verbose:
            print(f"[seed] Spec is fully balanced; using fallback a={a_fallback} with ell={ell_dom}. "
                  f"Seed beta=-pi/4, gamma={g1:.3e}")

    # Initialize params
    gen = torch.Generator().manual_seed(seed)
    # Initialize gammas around g1, betas around -pi/4
    gammas0 = torch.full((p,), float(g1), dtype=torch.get_default_dtype()) \
          + 0.2 * torch.rand((p,), generator=gen, dtype=torch.get_default_dtype()) * float(g1)
    betas0  = torch.full((p,), float(-np.pi/4), dtype=torch.get_default_dtype()) \
              + 0.05 * (torch.rand((p,), generator=gen, dtype=torch.get_default_dtype()) - 0.5)

    params = torch.cat([gammas0, betas0]).detach().clone().requires_grad_(True)


    # Freeze masks
    freeze_mask = torch.zeros(2*p, dtype=torch.bool)
    if mode == "fix-first":
        freeze_mask[0]   = True       # gamma_1
        freeze_mask[p+0] = True       # beta_1
    elif mode == "fix-first-two":
        if p < 3:
            raise ValueError("fix-first-two requires p >= 3")
        freeze_mask[0]     = True
        freeze_mask[p+0]   = True
        freeze_mask[1]     = True
        freeze_mask[p+1]   = True
    elif mode != "independent":
        raise ValueError("mode must be 'independent', 'fix-first', or 'fix-first-two'")

    # Optimizer
    opt = torch.optim.Adam([params], lr=lr)
    best_val = -1.0
    best_params = None

    trace = []

    for it in range(1, steps+1):
        # clamp betas
        _project_betas_(params, p)

        opt.zero_grad(set_to_none=True)
        success = circuit(params)

        loss = -success
        loss.backward()

        # Zero grads for frozen entries
        with torch.no_grad():
            if params.grad is not None:
                params.grad[freeze_mask] = 0.0

        opt.step()

        # Track the best params
        val = float(success.detach().cpu()) # detach() DOES exist unlike what PyCharm suggests.
        # pycharm cannot see that success is a torch.Tensor. but it is: interface="torch" in qnode
        trace.append(val)
        if val > best_val:
            best_val = val
            best_params = params.detach().clone()

        if verbose and (it % max(1, steps//6) == 0 or it == 1):
            print(f"[{it:4d}] success ~ {val:.6f}")

    # Final eval
    with torch.no_grad():
        final_probs = circuit(best_params)

    return {
        "best_success": best_val,
        "best_params": best_params,   # torch tensor length 2p: [gammas..., betas...]
        "final_probs": final_probs,   # torch scalar
        "trace": trace,
    }


def append_jsonl(path, obj):
    def _default(o):
        import torch, numpy as _np
        if isinstance(o, torch.Tensor): return o.detach().cpu().tolist()
        if isinstance(o, _np.ndarray):  return o.tolist()
        raise TypeError
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=_default) + "\n")  # no fsync


problem_specs = [
    [(1,1),(3,1),(4,1)],
    [(4,2)],
    [(1,1),(4,1)],
    [(3,1)],
    [(1,1),(3,1)],
    [(3,2)],
    [(4,2),(2,1)],
    [(2,1)],
    [(1,1),(3,2)],
    [(1,1),(4,2)]
]


def spec_max_ell(spec): return max(t[0] for t in spec)

# The task generator mirrors your nested loops
def make_tasks():
    runs = [
        (2, "independent"),
        (2, "fix-first"),
        (3, "independent"),
        (3, "fix-first"),
        (3, "fix-first-two"),
    ]
    for spec in problem_specs:
        for n in range(5, 13):
            if spec_max_ell(spec) > n:
                # encode a skip record so we can log it consistently (optional)
                yield ("__skip__", {"spec": spec, "n": n})
                continue
            for idx, (p, mode) in enumerate(runs):
                task = {
                    "spec": spec,
                    "n": n,
                    "p": p,
                    "mode": mode,
                    "seed": seed_base + idx,
                    "steps": 150,
                    "lr": 0.01,
                }
                yield ("run", task)

def run_one(task):
    """
    Worker function executed in a separate process.
    Returns a dict 'record' to be logged by the parent.
    """
    import time as _time

    # Reduce thread oversubscription inside each process
    try:
        torch.set_num_threads(1)
    except Exception as e:
        print(f"[warn] torch.set_num_threads(1) failed: {e}")

    import os
    try:
        for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
            os.environ[var] = "1"
    except Exception as e:
        print(f"[warn] setting env var failed: {e}")


    spec = task["spec"]; n = task["n"]; p = task["p"]; mode = task["mode"]
    steps = task["steps"]; lr = task["lr"]; seed = task["seed"]

    try:
        csp = SymmetricCSP(n=n, spec=spec, include_permutations=True, planted=[0]*n)

        # Force CPU (gpu unreliable)
        res = train_qaoa_layered_torch(
            csp, p=p, mode=mode,
            steps=steps, lr=lr, seed=seed,
            device_kind="lightning.qubit",
            dev_kwargs={},
            verbose=False,
        )
        record = {
            "timestamp": _time.time(),
            "spec": spec,
            "n": n,
            "p": p,
            "mode": mode,
            "best_success": res["best_success"],
            "best_params": res["best_params"],
            "trace": res["trace"],
        }
    except Exception as e:
        record = {
            "timestamp": _time.time(),
            "spec": spec,
            "n": n,
            "p": p,
            "mode": mode,
            "error": repr(e),
        }
    return record



def main(max_workers=None, run_id=None):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp

    # make a unique run_id per call
    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"qaoa_results_{run_id}.jsonl")

    # update the module-level seed_base so make_tasks() sees a new base each run
    global seed_base
    seed_base = int(run_id[-6:])  # last 6 digits of timestamp+usec

    # Safer start method for libraries using native threads
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


    tasks = list(make_tasks())
    # ensure dir exists
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)

    # submit only actual run tasks to the pool
    run_items = [(i, t) for i, (kind, t) in enumerate(tasks) if kind == "run"]
    skip_items = [(i, t) for i, (kind, t) in enumerate(tasks) if kind == "__skip__"]

    for _, meta in skip_items:
        print(f"skip spec={meta['spec']} at n={meta['n']} (ell_max > n)")

    # Parallel execution
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(run_one, t): (i, t) for (i, t) in run_items}
        for fut in as_completed(futs):
            i, t = futs[fut]
            try:
                record = fut.result()
            except Exception as e:
                # shouldn't happen
                record = {
                    "timestamp": time.time(),
                    "spec": t["spec"],
                    "n": t["n"],
                    "p": t["p"],
                    "mode": t["mode"],
                    "error": f"FutureError:{repr(e)}",
                }
            append_jsonl(out_file, record)
            print(f"logged: spec={record.get('spec')}, n={record.get('n')}, p={record.get('p')}, mode={record.get('mode')}")

if __name__ == "__main__":
    for i in range(10000):
        rid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{i:02d}"
        main(run_id=rid)
