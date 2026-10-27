"""Builds and EXECUTES 05_dirac_spinors.ipynb (no third-party deps).

The notebook is the executed companion of the ThotBook AI paper's Dirac
section: it verifies the Clifford algebra, the plane-wave spinors, the
stationary 1+1D solution that tab:pinn's PINN is graded against, and an
independent numerical dispersion check of the Dirac Hamiltonian.
"""
import sys

from nblite import build

# ---------------------------------------------------------------------------
CELLS = []
md = lambda s: CELLS.append(("md", s))
code = lambda s: CELLS.append(("code", s))

md(r"""# The Dirac Equation Worked Out in Spinors

**Goal.** Supply the fully-worked analytic skeleton behind the paper's PINN
benchmark (`tab:pinn`, Dirac 1+1D, graded rel-$L^2$ $1.7\times10^{-3}$ against
the exact $\cos kx$, $-(k/(E+m))\sin kx$ reference).  Everything here is
plain NumPy in float64 — no PyTorch, no SciPy.

In four steps:

1. **Clifford algebra** — the $\gamma^\mu$ matrices of the Dirac–Pauli basis
   and the 16-element verification of $\{\gamma^\mu,\gamma^\nu\}=2\eta^{\mu\nu}\mathbf 1$.
2. **Plane-wave spinors** — $u_s(p)$ normalisation and the on-shell identity
   $(\slashed p - m)\,u = 0$.
3. **The stationary 1+1D system actually solved by the PINN** — the
   $\alpha=\sigma_x$, $\beta=\sigma_z$ reduction, its eigenspinor ratio
   $\varphi_2/\varphi_1 = k/(E+m)$, and the exact real-space solution.
4. **Independent numerical dispersion check** — a finite-difference Dirac
   Hamiltonian on a periodic grid whose spectrum must reproduce
   $E(k)=\pm\sqrt{k^2+m^2}$ including the negative-energy branch.
""")

code(r"""import os
for _var in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
             "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np

np.set_printoptions(precision=4, suppress=True)
print("BLAS threading pinned to 1 before import: single-task-per-core is the measured regime")

# --- gamma matrices, Dirac-Pauli basis --------------------------------------
s1 = np.array([[0, 1], [1, 0]], dtype=complex)   # sigma_x
s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)  # sigma_y
s3 = np.array([[1, 0], [0, -1]], dtype=complex)   # sigma_z
I2 = np.eye(2, dtype=complex)

sigma = [s1, s2, s3]
g0 = np.diag([1, 1, -1, -1]).astype(complex)
g = [g0] + [np.block([[np.zeros((2, 2)), s], [-s, np.zeros((2, 2))]])
            for s in sigma]

eta = np.diag([1, -1, -1, -1]).astype(complex)

dev = 0.0
for mu in range(4):
    for nu in range(4):
        anti = g[mu] @ g[nu] + g[nu] @ g[mu]
        rhs = (2.0 * eta[mu, nu]) * np.eye(4, dtype=complex)
        dev = max(dev, float(np.max(np.abs(anti - rhs))))
print("Clifford {gm,gn} = 2 eta^{mn} 1  :  max deviation =", f"{dev:.2e}")

g5 = 1j * g[0] @ g[1] @ g[2] @ g[3]
print("g5^2 == 1                     :", float(np.max(np.abs(g5 @ g5 - np.eye(4)))) == 0.0)
anti5 = max(float(np.max(np.abs(g5 @ g[mu] + g[mu] @ g5))) for mu in range(4))
print("g5 anticommutes with all gm   :", f"{anti5:.2e}")
""")

md(r"""### 2. Plane-wave spinors and their exact normalisation

For $\slashed p = \gamma^\mu p_\mu$ and $E=\sqrt{\mathbf p^2+m^2}$, the
positive-energy spinors in the Dirac–Pauli basis are

$$u_s(p) \;=\; \begin{pmatrix} \sqrt{E+m}\,\chi_s \\[2pt]
\dfrac{\boldsymbol\sigma\cdot\mathbf p}{\sqrt{E+m}}\chi_s \end{pmatrix},
\qquad \chi_+ = \begin{pmatrix}1\\0\end{pmatrix},\enspace
\chi_- = \begin{pmatrix}0\\1\end{pmatrix}.$$

Three closed-form identities must hold *exactly* (to float64 round-off):

$$u^\dagger u = 2E, \qquad \bar u u = 2m \ \ (\bar u = u^\dagger \gamma^0),
\qquad (\slashed p - m\,\mathbf 1)\,u = 0 .$$
""")

code(r"""m = 1.0
pz = 1.4
E = float(np.sqrt(pz ** 2 + m ** 2))
s_dot_p = pz * s3                      # momentum along z

spinors = {}
for sidx, chi in enumerate((np.array([1, 0], dtype=complex),
                            np.array([0, 1], dtype=complex))):
    top = np.sqrt(E + m) * chi
    bot = ((s_dot_p / np.sqrt(E + m)) @ chi).reshape(2)
    u = np.concatenate([top, bot])
    spinors[sidx] = u

pslash = g0 * E - g[3] * pz            # gamma^0 E - gamma^3 pz
print(f"E = {E:.6f}   (pz = {pz}, m = {m})")
print(f"{'s':>2} {'u+u  = 2E':>18} {'ubar u = 2m':>14} {'|(pslash-m)u|':>16}")
for sidx, u in spinors.items():
    u_dag_u = float(np.real(u.conj() @ u))
    ubar = u.conj() @ g0
    ubar_u = float(np.real(ubar @ u))
    resid = float(np.linalg.norm(pslash @ u - m * u))
    print(f"{sidx:>2} {u_dag_u:>18.10f} {ubar_u:>14.10f} {resid:>16.2e}")
print(f"2E = {2*E:.10f}    2m = {2*m:.10f}")
""")

md(r"""### 3. The stationary 1+1D system the PINN actually solves

With $\alpha=\sigma_x$, $\beta=\sigma_z$, the stationary Dirac equation
$(\alpha p + \beta m)\,\psi = E\,\psi$ acting on a plane wave $e^{ikx}$ gives
the coupled component equations

$$k\,\varphi_2 + m\,\varphi_1 = E\,\varphi_1, \qquad
  k\,\varphi_1 - m\,\varphi_2 = E\,\varphi_2 ,$$

so the eigenspinor ratio is $\varphi_2/\varphi_1 = k/(E+m)$ and the
dispersion $E^2 = k^2 + m^2$ follows by multiplying the two equations.  The
published benchmark case is $E=2$, $m=1$, so $k=\sqrt 3 \approx 1.732051$ and
$\varphi_2/\varphi_1 = \sqrt3/2 \approx 0.866025$; the graded real-space
reference is $u(x)=\cos kx$, $v(x)=-(k/(E+m))\sin kx$ (leading real quadrature
of the eigenspinor pair).  Numerically: the $2\times 2$ eigensystem must
return the ratio to round-off, and the closed-form solution must satisfy the
coupled first-order system on a dense grid to round-off.
""")

code(r"""E, m = 2.0, 1.0
k = float(np.sqrt(E ** 2 - m ** 2))
H2d = np.block([[m * np.eye(1), np.array([[k]])],
                [np.array([[k]]), -m * np.eye(1)]])   # alpha*k + beta*m
vals, vecs = np.linalg.eigh(H2d)                       # symmetric: exact
ratio_num = float((vecs[:, 1][1] / vecs[:, 1][0]).real)   # E = +2 branch
ratio_exact = k / (E + m)
print(f"E = {E},  m = {m},  k = {k:.9f}   [sqrt(3) = {np.sqrt(3):.9f}]")
print(f"eigensystem eigenvalues (order +sqrt(k^2+m^2)) : {vals}  -> +- 2 as required")
print(f"eigenspinor ratio (E=+2 branch) : {ratio_num:.15f}")
print(f"analytic ratio   k/(E+m)        : {ratio_exact:.15f}")
print(f"deviation                       : {abs(ratio_num - ratio_exact):.2e}")

# exact real-space solution residual on a dense grid
x = np.linspace(0.0, 6.0, 2001)
u = np.cos(k * x)
v = -(k / (E + m)) * np.sin(k * x)
du = np.gradient(u, x)                       # FD derivative is O(h^2)
dv = np.gradient(v, x)
r1 = du - (E + m) * v                        # du/dx = +(E+m) v
r2 = dv + (E - m) * u                        # dv/dx = -(E-m) u
print(f"closed-form solution residual (O(h^2) FD derivative): "
      f"max|r1| = {np.max(np.abs(r1)):.2e},  max|r2| = {np.max(np.abs(r2)):.2e}")
""")

md(r"""### 4. Independent numerical check: the spectrum of the Dirac Hamiltonian

The first-order Dirac Hamiltonian $H = -i\,\alpha\,\partial_x + \beta\,m$ on a
periodic grid is discretised with a central difference ($D$ antisymmetric, so
$-iD$ is Hermitian — no stabilisation tricks), giving a $2N\times 2N$ Hermitian
eigenproblem.  The *discrete* dispersion replaces $k$ by $\sin(kh)/h$ (the
central-difference momentum, correct to $O(h^2)$), so the exact statement the
spectral check must satisfy at float64 round-off is

$$E_\pm(\tilde k_j) = \pm\sqrt{\tilde k_j^2+m^2},\qquad
\tilde k_j=\tfrac{1}{h}\sin\tfrac{2\pi j}{N},$$

for **both** branches: the negative-energy branch (the Dirac sea, and with it
antimatter) appears as a *derived* spectral fact, not a postulate.
""")

code(r"""L, N = 12.0, 512
h = L / N
D = np.zeros((N, N))
for j in range(N):
    D[j, (j + 1) % N] = 1.0
    D[j, (j - 1) % N] = -1.0
D /= (2.0 * h)                               # antisymmetric central difference
neg_iD = -1j * D                             # exactly Hermitian
I_grid = np.eye(N)
# H = alpha (-i d/dx) + beta m in the (u, v) basis:
H = np.zeros((2 * N, 2 * N), dtype=complex)
H[:N, :N] = m * I_grid                       # beta = sigma_z: +m on the u block
H[N:, N:] = -m * I_grid                      #                  -m on the v block
H[:N, N:] = neg_iD                           # alpha = sigma_x off-diagonal
H[N:, :N] = neg_iD
vals_sorted = np.sort(np.linalg.eigvalsh(H))

# exact DISCRETE dispersion: eigenvalue of -iD on mode j is sin(theta_j)/h
theta = 2.0 * np.pi * np.arange(N) / N
kdisc = np.sin(theta) / h
Etheory = np.sort(np.concatenate([np.sqrt(kdisc ** 2 + m ** 2),
                                  -np.sqrt(kdisc ** 2 + m ** 2)]))
dev = float(np.max(np.abs(vals_sorted - Etheory)))
print(f"grid N={N}, L={L}, h={h:.4f}: FD spectrum vs discrete dispersion +-sqrt(k~^2+m^2)")
print(f"max |spectrum discrepancy| = {dev:.2e}   (round-off, as it must be)")
print(f"lowest eigenvalue (Dirac sea)  : {vals_sorted[0]:.9f}")
print(f"highest eigenvalue             : {vals_sorted[-1]:.9f}")
print(f"pair symmetry max |E_j + E_-j| : "
      f"{np.max(np.abs(vals_sorted[:N] + vals_sorted[N:][::-1])):.2e}")
""")

md(r"""### Verification summary

| Quantity checked | Expected | Measured |
|---|---|---|
| Clifford anticommutators (16 pairs) | 0 deviation | round-off |
| $u^\dagger u$, $\bar u u$, $(\slashed p-m)u=0$ | $2E$, $2m$, 0 | round-off |
| eigenspinor ratio $\varphi_2/\varphi_1$ | $k/(E+m)=\sqrt3/2$ | round-off |
| closed-form $\cos kx$/$\sin kx$ residual | $O(h^2)$ only | $O(h^2)$ |
| FD Dirac spectrum vs $\pm\sqrt{k^2+m^2}$ | round-off | round-off |

The analytic spinor solution is the closed-form grading target of the PINN
benchmark in the paper — no new empirical claims are made here.
""")

build(sys.argv[1] if len(sys.argv) > 1 else "05_dirac_spinors.ipynb", CELLS)
