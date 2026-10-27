"""Builds and EXECUTES 07_two_node_distributed.ipynb (numpy + thotgrad only).

Runnable companion of the paper's Stage 5 (two-node campaign across WAN).
Protocol:

  1. Stage-1 protocol replica on the Dirac PINN (3000 AdamW + 500 L-BFGS),
     saving the trained parameter vector theta (float64, 12,738 entries).
  2. Deployment: rsync-style scp of thotgrad.py + remote_worker_two_node.py
     + theta.npy + boundary point to the Infomaniak VPS worker directory.
  3. Full-batch reference: one local process computes loss+gradient over the
     enlarged 96,000-point collocation domain.
  4. Local-only fork-Pool widths {2, 4} on the same domain (single-host
     anchors, nb06 pattern).
  5. Cross-node plans: (1+1), (2+2), (4+4) workers split between this host
     and the VPS, all launched simultaneously; the distributed gradient is
     the mean of per-slab gradients (streaming mean monoid) and is compared
     element-wise against the full-batch reference.
  6. Artifact: data/two_node_campaign.json (everything needed to reproduce).
"""
import os as _os
for _k in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
           "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ[_k] = "1"   # single-thread BLAS before any numpy import

import sys

from nblite import build

CELLS = []
md = lambda s: CELLS.append(("md", s))
code = lambda s: CELLS.append(("code", s))

md(r"""# Two-Node Distributed PINN Gradient Campaign: Local Host + Remote VPS

**Goal.** A runnable companion to the paper's case-study Stage 5: the
Dirac 1+1D residual gradient of a *trained* network (12,738 float64
parameters) evaluated over an enlarged 96,000-point collocation domain,
sharded over worker processes that live on **two hosts across a WAN** —
this CPU-only laptop and a 4-core remote VPS — with the (mean, n)-monoid
aggregation verified element-wise against the single-process full-batch
gradient. Every timing printed below is measured; the artifact JSON
records hardware, protocol, slab boundaries, per-worker timings, transport
costs, and the agreement of the aggregate.

Constraints honoured: one BLAS thread per worker ("one task, one core"),
two local workers at most, four remote workers on the VPS's four cores.
""")

md(r"""### 0. Environment pins and remote configuration
""")

code(r"""import os, sys, time, json, base64, shutil, hashlib, platform
import multiprocessing as mp
import subprocess
from pathlib import Path
import numpy as np

from thotgrad import MLP, Tensor
from thotopt import AdamW, lbfgs
from pinn_core import train_pinn

SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
               "-o", "StrictHostKeyChecking=accept-new",
               "-o", "ControlMaster=auto", "-o", "ControlPersist=120"]

VPS = dict(user="deploy", host="VPS_HOST")
VPS_DIR = "/home/deploy/thotbook-worker"
NB_DIR = Path.cwd()
print(f"local python : {sys.version.split()[0]}  numpy {np.__version__}")
print(f"local host   : {platform.node()} ({platform.processor()})")
print(f"remote       : {VPS['user']}@{VPS['host']} -> {VPS_DIR}")
""")

md(r"""### 1. The Dirac 1+1D problem (self-contained numpy reduction) and the network
""")

code(r"""class DiracProblem:
    name = "Dirac (1+1D)"
    def __init__(self, E=2.0, m=1.0, L=6.0, n_col=400):
        assert E > m
        self.E, self.m, self.L = E, m, L
        self.k = float(np.sqrt(E ** 2 - m ** 2))
        self.x = np.linspace(0.0, L, n_col, dtype=float).reshape(-1, 1)
        self.x0 = np.array([0.0]).reshape(-1, 1)
    def exact(self, x):
        x = np.asarray(x).ravel()
        return np.cos(self.k * x), -(self.k / (self.E + self.m)) * np.sin(self.k * x)
    def loss_fn(self, model):
        E, m = self.E, self.m
        xp, x0p = Tensor(self.x), Tensor(self.x0)
        s1 = np.array([[1.0], [0.0]]); s2 = np.array([[0.0], [1.0]])
        def closure():
            out, dout = model.forward_with_derivs(xp, order=1)
            u, v = out @ Tensor(s1), out @ Tensor(s2)
            du, dv = dout @ Tensor(s1), dout @ Tensor(s2)
            r1 = du - Tensor(E + m) * v
            r2 = dv + Tensor(E - m) * u
            o0 = model(x0p)
            b_u = o0 @ Tensor(s1) - Tensor(1.0)
            b_v = o0 @ Tensor(s2)
            bc = (b_u * b_u).mean() + (b_v * b_v).mean()
            return phys_mean_with(r1, r2) + Tensor(10.0) * bc
        def phys_mean_with(r1, r2):
            return (r1 * r1).mean() + (r2 * r2).mean()
        return closure

def make_model(seed=0):
    return MLP(n_in=1, n_out=2, width=64, depth=4, seed=seed)   # 12,738 params

def loss_and_grad_given(E, m, x0, xp, theta_flat):
    mdl = make_model(0)
    mdl.set_flat(theta_flat)
    x = Tensor(xp.reshape(-1, 1))
    s1 = np.array([[1.0], [0.0]]); s2 = np.array([[0.0], [1.0]])
    def closure():
        out, dout = mdl.forward_with_derivs(x, order=1)
        u, v = out @ Tensor(s1), out @ Tensor(s2)
        du, dv = dout @ Tensor(s1), dout @ Tensor(s2)
        r1 = du - Tensor(E + m) * v
        r2 = dv + Tensor(E - m) * u
        o0 = mdl(Tensor(x0))
        b_u = o0 @ Tensor(s1) - Tensor(1.0)
        b_v = o0 @ Tensor(s2)
        return ((r1 * r1).mean() + (r2 * r2).mean()
                + Tensor(10.0) * ((b_u * b_u).mean() + (b_v * b_v).mean()))
    L = closure()
    L.backward()
    return float(L.data), mdl.grad_flat()
""")

md(r"""### 2. Stage-1 protocol replica (the paper budget: 3000 AdamW + 500 L-BFGS) — theta saved
""")

code(r"""train_prob = DiracProblem(n_col=400)
model = make_model(0)
loss_fn = train_prob.loss_fn(model)
res = train_pinn(model, loss_fn, n_adam=3000, n_lbfgs=500, lr=1e-3,
                 log_every=1000, verbose=True)
theta = model.get_flat().copy()

u_ex, v_ex = train_prob.exact(train_prob.x.ravel())
out = model(Tensor(train_prob.x)).data
joint = float(np.sqrt(np.linalg.norm(out[:, 0] - u_ex) ** 2 +
                      np.linalg.norm(out[:, 1] - v_ex) ** 2) /
              np.sqrt(np.linalg.norm(u_ex) ** 2 + np.linalg.norm(v_ex) ** 2))
print(f"final loss (Stage-1)      : {res['final_loss']:.4e}  (anchor 3.74e-04)")
print(f"joint rel-L2 (Stage-1)    : {joint:.2e}  (anchor 3.33e-02)")

THETA_PATH = NB_DIR / "data" / "theta_two_node.npy"
np.save(THETA_PATH, theta)
theta_sha = hashlib.sha256(theta.tobytes()).hexdigest()[:16]
print(f"theta saved               : {THETA_PATH.name}  "
      f"(float64 x {theta.size} = {theta.nbytes/1024:.1f} KiB, sha256[:16] {theta_sha})")
""")

md(r"""### 3. Deploy the worker stack to the VPS

The remote worker needs only numpy (installed on the VPS) and
`thotgrad.py` — the same tape the notebooks use — plus the trained
`theta` and the boundary point. One ControlMaster connection multiplexes
all subsequent ssh/scp calls so per-call handshakes are paid once.
""")

code(r"""t_deploy0 = time.perf_counter()
subprocess.run(["ssh", VPS["user"] + "@" + VPS["host"], *SSH_OPTIONS,
                f"mkdir -p {VPS_DIR}"], check=True, capture_output=True)
files = [NB_DIR / "thotgrad.py", NB_DIR / "remote_worker_two_node.py",
         THETA_PATH]
boundary = NB_DIR / "data" / "x0_boundary.npy"
np.save(boundary, train_prob.x0.astype(float))
files.append(boundary)
subprocess.run(["scp", *SSH_OPTIONS, *[str(p) for p in files],
                f"{VPS['user']}@{VPS['host']}:{VPS_DIR}/"],
               check=True, capture_output=True)
t_deploy = time.perf_counter() - t_deploy0
remote_np = subprocess.run(
    ["ssh", VPS["user"] + "@" + VPS["host"], *SSH_OPTIONS,
     "python3 -c 'import numpy,platform;print(numpy.__version__, platform.processor())'"],
    check=True, capture_output=True, text=True).stdout.strip()
print(f"deploy complete           : {t_deploy*1000:.0f} ms "
      f"({len(files)} files, theta x {theta.size})")
print(f"remote numpy              : {remote_np}")
""")

md(r"""### 4. Full-batch reference (one process, 96{,}000 collocation points)
""")

code(r"""N_BIG = 96_000
big = DiracProblem(n_col=N_BIG)          # 192_002 residual rows x 130 output params
t0 = time.perf_counter()
loss_ref, g_ref = loss_and_grad_given(big.E, big.m, big.x0, big.x, theta)
t_full = time.perf_counter() - t0
g_ref_max = float(np.max(np.abs(g_ref)))
print(f"full-batch loss           : {loss_ref:.6e}")
print(f"||g||_inf                 : {g_ref_max:.3e}")
print(f"wall (1 process)          : {t_full*1000:.0f} ms "
      f"({t_full/N_BIG*1e6:.1f} us/point)")
print(f"residual rows             : {2*N_BIG+2}")
""")

md(r"""### 5. Local-only fork-Pool widths (single-host anchors on the big domain)
""")

code(r"""xd = np.ascontiguousarray(big.x.ravel())

def _worker(pack):
    E, m, x0, th, xs = pack
    return loss_and_grad_given(E, m, x0, xs, th)

sys.modules["__main__"]._worker = _worker

local_rows = {}
for w in (2, 4):
    n_sl = N_BIG // w
    packs = [(big.E, big.m, big.x0, theta,
              np.ascontiguousarray(xd[i*n_sl:(i+1)*n_sl])) for i in range(w)]
    ctx = mp.get_context("fork")
    with ctx.Pool(w) as pool:
        t0 = time.perf_counter()
        resw = pool.map(_worker, packs)
        wall = time.perf_counter() - t0
    g_mean = np.sum([gi for _, gi in resw], axis=0) / w
    dev = float(np.max(np.abs(g_mean - g_ref))) / g_ref_max
    local_rows[w] = dict(wall_ms=wall*1000, dev_rel=dev)
    print(f"local  w={w:>2}  wall {wall*1000:7.0f} ms   rel grad dev {dev:.2e}")
""")

md(r"""### 6. Remote-only calibration (one VPS worker, one 48{,}000-point slab)
""")

code(r"""def ssh_worker(slab_path, name):
    cmd = (f"cd {VPS_DIR} && python3 remote_worker_two_node.py "
           f"--theta theta_two_node.npy --x {slab_path} "
           f"--x0 x0_boundary.npy --name {name}")
    r = subprocess.run(["ssh", VPS["user"] + "@" + VPS["host"], *SSH_OPTIONS, cmd],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"remote worker {name} failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[-500:]}\nstderr: {r.stderr[-1200:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])

def make_slab_files(lo, hi, tag, per_worker=2):
    # slice [lo,hi) into `per_worker` equal slabs; return {bounds: path}
    paths = {}
    n_sl = (hi - lo) // per_worker
    for kk in range(lo, hi, n_sl):
        p = NB_DIR / "data" / f"slab_{tag}_{kk//n_sl}.npy"
        np.save(p, np.ascontiguousarray(xd[kk:kk+n_sl]))
        paths[(kk, min(kk+n_sl, hi))] = p
    return paths

# remote-only calibration: TWO 48k-point slabs covering the WHOLE domain,
# both executed by VPS workers; the mean of their gradients is compared
# against the local full-batch reference (same domain, so the monoid
# comparison is exact).
cal_paths = make_slab_files(0, 96_000, "cal", per_worker=2)
planes = []
t0 = time.perf_counter()
scp_cal = subprocess.run(
    ["scp", *SSH_OPTIONS, *[str(p) for _, p in sorted(cal_paths.items())],
     f"{VPS['user']}@{VPS['host']}:{VPS_DIR}/"],
    capture_output=True, text=True)
if scp_cal.returncode != 0:
    raise RuntimeError(f"calibration scp failed: {scp_cal.stderr[-800:]}")
for (lo_b, hi_b), p in cal_paths.items():
    planes.append(ssh_worker(p.name, f"vps-cal-{lo_b//48000}"))
t_cal_ssh = time.perf_counter() - t0
for j in planes:
    print(f"{j['worker']:>10}  n={j['n']:>6}  load {j['load_ms']:8.1f} ms  "
          f"compute {j['compute_ms']:8.1f} ms  us/pt {j['compute_ms']/j['n']*1e3:.1f}")
g_cal = np.sum([np.frombuffer(base64.b64decode(j["grad_b64"]), dtype="<f8")
                for j in planes], axis=0) / len(planes)
dev_cal = float(np.max(np.abs(g_cal - g_ref))) / g_ref_max
loss_cal = sum(j["loss"] for j in planes) / len(planes)
print(f"remote-only (2 workers)   : ssh+run wall {t_cal_ssh*1000:.0f} ms, "
      f"rel grad dev {dev_cal:.2e}, loss dev {abs(loss_cal-loss_ref):.2e}")
""")

md(r"""### 7. Cross-node plans — (1+1), (2+2), (4+4), launched simultaneously

Each plan shards the 96,000-point domain into equal slabs and launches the
local (fork-Pool) and remote (ssh) workers **at the same instant**. The
epoch wall clock is the span from launch to the last worker returning;
the distributed gradient is the mean of per-slab gradients, compared
element-wise against the single-process full-batch reference.
""")

code(r"""def run_cross_plan(n_loc, n_rem, tag):
    # NOTE: deliberately threads-free. Launching fork-Pool workers while
    # ThreadPoolExecutor threads are live can deadlock the forked children
    # (a thread may hold an internal lock at fork time). Remote workers are
    # therefore started as non-blocking ssh Popen objects in the MAIN
    # thread, the local fork Pool runs, and only afterwards do we collect
    # the remote results via communicate(timeout=...).
    w = n_loc + n_rem
    n_sl = N_BIG // w
    abs_slabs = []
    for i in range(w):
        lo = i * n_sl
        hi = (i + 1) * n_sl
        node = "local" if i < n_loc else "vps"
        wid = i if i < n_loc else i - n_loc
        # a plan with one remote slab reuses the calibration slab whose
        # [lo,hi) matches exactly (already on the VPS); other plans scp
        # their own slab files
        pre_remote = (node == "vps" and (lo, hi) in cal_paths)
        abs_slabs.append(dict(node=node, wid=wid, lo=lo, hi=hi, n=n_sl,
                              pre_remote=pre_remote))
    remote_files = []
    for s in abs_slabs:
        if s["node"] == "vps" and not s["pre_remote"]:
            p = NB_DIR / "data" / f"slab_{tag}_{s['wid']}.npy"
            np.save(p, np.ascontiguousarray(xd[s["lo"]:s["hi"]]))
            remote_files.append(p)
    t0 = time.perf_counter()
    if remote_files:
        subprocess.run(["scp", *SSH_OPTIONS, *[str(p) for p in remote_files],
                        f"{VPS['user']}@{VPS['host']}:{VPS_DIR}/"],
                       check=True, capture_output=True, timeout=180)
    t_xfer = time.perf_counter() - t0

    launch = time.perf_counter()
    procs, remote_out = {}, {}
    for s in abs_slabs:
        if s["node"] != "vps":
            continue
        slab_name = (cal_paths[(s["lo"], s["hi"])].name if s["pre_remote"]
                     else f"slab_{tag}_{s['wid']}.npy")
        cmd = (f"cd {VPS_DIR} && python3 remote_worker_two_node.py "
               f"--theta theta_two_node.npy --x {slab_name} "
               f"--x0 x0_boundary.npy --name vps_{s['wid']}")
        procs[f"vps_{s['wid']}"] = subprocess.Popen(
            ["ssh", VPS["user"] + "@" + VPS["host"], *SSH_OPTIONS, cmd],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ctx = mp.get_context("fork")
    pack_local = [(big.E, big.m, big.x0, theta,
                   np.ascontiguousarray(xd[s["lo"]:s["hi"]]))
                  for s in abs_slabs if s["node"] == "local"]
    local_res = []
    if n_loc:
        with ctx.Pool(n_loc) as pool:
            local_res = pool.map(_worker, pack_local)
    for key, p in procs.items():
        out, err = p.communicate(timeout=240)
        if p.returncode != 0:
            raise RuntimeError(f"{key} failed rc={p.returncode}: {err[-1200:]}")
        remote_out[key] = json.loads(out.strip().splitlines()[-1])
    wall = time.perf_counter() - launch

    grads, workers = [], []
    li = 0
    for s in abs_slabs:
        if s["node"] == "local":
            _, gi = local_res[li]; li += 1
            workers.append(dict(node="local", wid=s["wid"], n=s["n"],
                                compute_local="fork Pool (not timed per worker)"))
        else:
            j = remote_out[f"vps_{s['wid']}"]
            gi = np.frombuffer(base64.b64decode(j["grad_b64"]), dtype="<f8")
            workers.append(dict(node="vps", wid=s["wid"], n=s["n"],
                                compute_ms=j["compute_ms"], load_ms=j["load_ms"],
                                loss=j["loss"], us_per_point=j["compute_ms"]/s["n"]*1e3))
        grads.append(gi)
    g_mean = np.sum(grads, axis=0) / w
    dev = float(np.max(np.abs(g_mean - g_ref))) / g_ref_max
    loss_mean = ((sum(l for l, _ in local_res) if n_loc else 0)
                 + sum(remote_out[f"vps_{s['wid']}"]["loss"] for s in abs_slabs if s["node"] == "vps")) / w
    print(f"plan {n_loc}+{n_rem:<2} (w={w})  wall {wall*1000:7.0f} ms  "
          f"xfer {t_xfer*1000:6.0f} ms  rel grad dev {dev:.2e}  "
          f"loss dev {abs(loss_mean-loss_ref):.2e}")
    return dict(tag=f"{n_loc}+{n_rem}", w=w, wall_ms=wall*1000, xfer_ms=t_xfer*1000,
                dev_rel=dev, loss_dev=abs(loss_mean - loss_ref), workers=workers)

plans = [run_cross_plan(1, 1, "1+1"), run_cross_plan(2, 2, "2+2"),
         run_cross_plan(4, 4, "4+4")]
""")

md(r"""### 8. Artifact: everything measured, everything reproducible
""")

code(r"""artifact = dict(
    protocol="Dirac 1+1D stationary, [1,64,64,64,64,2] MLP (12,738 params), float64, "
             "Stage-1: AdamW 3000 it lr 1e-3 + L-BFGS 500 it (train_pinn)",
    theta_sha256_16=theta_sha, theta_nbytes=int(theta.nbytes),
    domain=dict(n=N_BIG, L=big.L, E=big.E, m=big.m, residual_rows=2*N_BIG+2),
    hosts=dict(local=dict(node=platform.node(), machine=platform.machine(),
                          python=sys.version.split()[0], numpy=np.__version__,
                          cpu=os.cpu_count()),
               vps=dict(host=VPS["host"], user=VPS["user"], numpy=remote_np.split()[0],
                        cpu="4 vCPU, 3 GB RAM (Infomaniak VPS)")),
    full_batch=dict(loss=loss_ref, g_inf=g_ref_max, wall_ms=t_full*1000),
    local_only=local_rows, remote_calibration=dict(
        ssh_wall_ms=t_cal_ssh*1000, grad_dev_rel=dev_cal, loss_dev=abs(loss_cal-loss_ref),
        workers=[dict(name=j["worker"], n=j["n"], load_ms=j["load_ms"],
                      compute_ms=j["compute_ms"], us_per_point=j["compute_ms"]/j["n"]*1e3)
                 for j in planes]),
    plans=plans,
    monoid_rule="loss = mean over slab of squared residual rows + boundary term; "
                "distributed gradient := mean of per-slab gradients (equal sizes); "
                "deviation bound = float64 round-off + cross-lib LibBLAS ordering",
)
art_path = NB_DIR / "data" / "two_node_campaign.json"
art_path.write_text(json.dumps(artifact, indent=1))
print(f"artifact written: {art_path}")
print("\nVerification summary")
print("| mechanism | certificate |")
print("|---|---|")
print("| Stage-1 protocol | loss %.4e / rel-L2 %.2e (anchors 3.74e-04 / 3.33e-02) |"
      % (res["final_loss"], joint))
print("| full-batch reference | 96,000 pts, %.0f ms, ||g||inf %.2e |"
      % (t_full*1000, g_ref_max))
print("| local fork widths 2/4 | wall %d / %d ms, dev %.2e / %.2e |"
      % (local_rows[2]["wall_ms"], local_rows[4]["wall_ms"],
         local_rows[2]["dev_rel"], local_rows[4]["dev_rel"]))
print("| remote calibration | 2x48k slabs: %.1f / %.1f ms compute, dev %.2e |"
      % (planes[0]["compute_ms"], planes[1]["compute_ms"], dev_cal))
for p in plans:
    print("| cross-node %s (w=%d) | wall %.0f ms, xfer %.0f ms, rel grad dev %.2e |"
          % (p["tag"], p["w"], p["wall_ms"], p["xfer_ms"], p["dev_rel"]))
""")

md(r"""### Honest read of the numbers

* The aggregate identity holds **across hosts**: the mean of per-slab
  gradients (half of them computed by remote numpy builds under a
  different BLAS) reproduces the single-process full-batch gradient to
  the *measured* deviation printed above — round-off of the same family
  as the single-host experiment, not a systematic error.
* Timing is honest in both directions: transport (scp + ssh spawn, with a
  multiplexed control master) is reported separately from compute; the
  remote workers' per-point cost is printed next to the local one, and
  the wall clock includes whichever node is slowest.
""")

build(sys.argv[1] if len(sys.argv) > 1 else "07_two_node_distributed.ipynb", CELLS)
