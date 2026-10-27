"""The three physics problems solved by PINNs in 04_pinn_physics.ipynb.

Each problem exposes:
    .residual(model, x)  -> PDE/ODE residual tensor(s)
    .exact(x)            -> reference solution (analytic, or high-accuracy ODE)
    .loss_fn(model)      -> closure for pinn_core.train_pinn

All three have an independently-checkable ground truth, so the PINN is never
graded against itself. Autodiff is `thotgrad` (this project); SciPy is used
only to *generate references*, never inside the solver — grading a solver with
itself would be circular.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import airy

from thotgrad import Tensor


def _col(x) -> Tensor:
    return Tensor(np.asarray(x, dtype=float).reshape(-1, 1))


# ---------------------------------------------------------------------------
# 1. DIRAC EQUATION (1+1 D, stationary, two-component spinor)
# ---------------------------------------------------------------------------
class DiracProblem:
    r"""Stationary Dirac equation in 1+1 dimensions.

    In the standard two-component reduction the stationary Dirac system is the
    coupled first-order pair

        du/dx = +(E + m) v
        dv/dx = -(E - m) u

    Differentiating once decouples it:  u'' = -(E^2 - m^2) u = -k^2 u,
    which is exactly the relativistic dispersion relation  E^2 = k^2 + m^2.

    With u(0) = 1, v(0) = 0 the closed-form solution is

        u(x) = cos(kx),      v(x) = -(k/(E+m)) sin(kx),     k = sqrt(E^2 - m^2)

    so the PINN can be graded against an exact analytic benchmark.
    """

    name = "Dirac (1+1D)"

    def __init__(self, E: float = 2.0, m: float = 1.0, L: float = 6.0,
                 n_col: int = 400):
        assert E > m, "need E > m for a propagating (real-k) solution"
        self.E, self.m, self.L = E, m, L
        self.k = float(np.sqrt(E ** 2 - m ** 2))
        self.x = _col(np.linspace(0.0, L, n_col))
        self.x0 = _col([0.0])

    def exact(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.cos(self.k * x)
        v = -(self.k / (self.E + self.m)) * np.sin(self.k * x)
        return u, v

    def loss_fn(self, model):
        def closure():
            out, dout = model.forward_with_derivs(self.x, order=1)
            u, v = _slice(out, 0), _slice(out, 1)
            du = _slice(dout, 0); dv = _slice(dout, 1)
            r1 = du - Tensor(self.E + self.m) * v
            r2 = dv + Tensor(self.E - self.m) * u
            phys = (r1 * r1).mean() + (r2 * r2).mean()
            o0 = model(self.x0)
            b_u = _slice(o0, 0) - Tensor(1.0)
            b_v = _slice(o0, 1)
            bc = (b_u * b_u).mean() + (b_v * b_v).mean()
            return phys + Tensor(10.0) * bc
        return closure


def _slice(t: Tensor, j: int) -> Tensor:
    """Differentiable column slice t[:, j:j+1]."""
    sel = np.zeros((t.data.shape[1], 1))
    sel[j, 0] = 1.0
    return t @ Tensor(sel)


# ---------------------------------------------------------------------------
# 2. BRANS-DICKE COSMOLOGY (flat FLRW + dust)
# ---------------------------------------------------------------------------
class BransDickeProblem:
    r"""Brans-Dicke scalar-tensor cosmology, flat FLRW with dust.

    Field equations (units 8*pi = c = 1), with H = adot/a and rho = rho0 a^-3:

        (F)  3H^2 = rho/phi + (omega/2)(phidot/phi)^2 - 3H(phidot/phi)
        (S)  phiddot + 3H phidot = rho/(2 omega + 3)

    The Nariai power-law solution is exact:

        a(t) = t^q,   q = (2 + 2 omega)/(4 + 3 omega)
        phi(t) = t^s, s = 2/(4 + 3 omega)

    consistent iff rho0 = phi0 * s * (2 omega + 3).  As omega -> infinity,
    q -> 2/3, recovering the GR dust limit -- the standard sanity check that
    Brans-Dicke reduces to general relativity.

    The network learns log a and log phi, which enforces positivity
    structurally; a and phi are recovered by exponentiation.
    """

    name = "Brans-Dicke (FLRW + dust)"

    def __init__(self, omega: float = 10.0, t0: float = 1.0, t1: float = 3.0,
                 n_col: int = 400):
        self.omega = omega
        D = 4.0 + 3.0 * omega
        self.q = (2.0 + 2.0 * omega) / D
        self.s = 2.0 / D
        self.rho0 = 1.0 * self.s * (2.0 * omega + 3.0)   # phi0 = 1
        self.t = _col(np.linspace(t0, t1, n_col))
        self.tic = _col([t0])
        self.t0 = t0

    def exact(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return t ** self.q, t ** self.s

    def loss_fn(self, model):
        w = self.omega

        def closure():
            out, dout, d2out = model.forward_with_derivs(self.t, order=2)
            la, lp = _slice(out, 0), _slice(out, 1)          # log a, log phi
            dla, dlp = _slice(dout, 0), _slice(dout, 1)
            d2lp = _slice(d2out, 1)

            # H = adot/a = d(log a)/dt ;  phidot/phi = d(log phi)/dt
            H = dla
            pr = dlp                                          # phidot/phi
            # phiddot/phi = (log phi)'' + ((log phi)')^2
            ppr = d2lp + dlp * dlp

            # rho/phi = rho0 * exp(-3 log a - log phi)
            rho_over_phi = Tensor(self.rho0) * _exp(Tensor(-3.0) * la - lp)

            rF = (Tensor(3.0) * H * H - rho_over_phi
                  - Tensor(0.5 * w) * pr * pr + Tensor(3.0) * H * pr)
            # (S) divided through by phi: phiddot/phi + 3H phidot/phi
            #                              = rho/((2w+3) phi)
            rS = ppr + Tensor(3.0) * H * pr - rho_over_phi / Tensor(2.0 * w + 3.0)

            phys = (rF * rF).mean() + (rS * rS).mean()
            o = model(self.tic)                # log a(t0)=0, log phi(t0)=0
            b0, b1 = _slice(o, 0), _slice(o, 1)
            bc = (b0 * b0).mean() + (b1 * b1).mean()
            return phys + Tensor(100.0) * bc
        return closure


def _exp(t: Tensor) -> Tensor:
    """exp via tanh-free identity is unavailable; add a dedicated node."""
    e = np.exp(t.data)
    out = Tensor(e, _children=(t,))

    def _bw():
        if t.requires_grad:
            t._ensure_grad()
            t.grad += out.grad * e
    out._backward = _bw
    return out


# ---------------------------------------------------------------------------
# 3. WHEELER-DEWITT EQUATION (minisuperspace, closed FLRW + Lambda)
# ---------------------------------------------------------------------------
class WheelerDeWittProblem:
    r"""Wheeler-DeWitt equation in closed-FLRW minisuperspace.

        Psi''(a) + (p/a) Psi'(a) - U(a) Psi(a) = 0,
        U(a) = a^2 - (Lambda/3) a^4

    p is the Hartle-Hawking factor-ordering parameter (the operator-ordering
    ambiguity of quantising a curved minisuperspace).  The classical turning
    point sits at a_t = sqrt(3/Lambda): U > 0 (barrier, exponential) for
    a < a_t, U < 0 (oscillatory, classically allowed) for a > a_t -- the
    tunnelling-from-nothing configuration.

    Two independent ground truths:
      * `exact` integrates the full ODE with SciPy's adaptive Radau solver at
        rtol=1e-12;
      * `airy_asymptotic` gives the analytic Airy form valid near a_t, where
        linearising U(a) ~ U'(a_t)(a - a_t) turns the equation into
        Psi_zz = z Psi.
    """

    name = "Wheeler-DeWitt (minisuperspace)"

    def __init__(self, lam: float = 1.0, p: float = 0.0, a_lo: float = 0.3,
                 a_hi: float = 2.6, n_col: int = 500):
        self.lam, self.p = lam, p
        self.a_lo, self.a_hi = a_lo, a_hi
        self.a_turn = float(np.sqrt(3.0 / lam))
        self.a_np = np.linspace(a_lo, a_hi, n_col)
        self.a = _col(self.a_np)
        self.aic = _col([a_lo])
        self.psi0, self.dpsi0 = 1.0, 0.0

    def U(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        return a ** 2 - (self.lam / 3.0) * a ** 4

    def exact(self, a_eval: np.ndarray) -> np.ndarray:
        """High-accuracy reference via adaptive implicit ODE integration."""
        def rhs(a, y):
            psi, dpsi = y
            return [dpsi, -(self.p / a) * dpsi + (a ** 2 - (self.lam / 3) * a ** 4) * psi]
        sol = solve_ivp(rhs, (self.a_lo, self.a_hi), [self.psi0, self.dpsi0],
                        t_eval=np.sort(a_eval), method="Radau",
                        rtol=1e-12, atol=1e-14)
        assert sol.success, sol.message
        return sol.y[0]

    def airy_asymptotic(self, a_eval: np.ndarray) -> np.ndarray:
        """Analytic Airy solution of the linearised equation near a_turn."""
        at = self.a_turn
        Uprime = 2 * at - (4 * self.lam / 3) * at ** 3      # = -2*sqrt(3) for lam=1
        c = abs(Uprime) ** (1.0 / 3.0)
        z = -c * (a_eval - at)
        return airy(z)[0]                                    # Ai(z)

    def loss_fn(self, model):
        Uvals = Tensor(self.U(self.a_np).reshape(-1, 1))
        inv_a = Tensor((self.p / self.a_np).reshape(-1, 1))

        def closure():
            psi, dpsi, d2psi = model.forward_with_derivs(self.a, order=2)
            r = d2psi + inv_a * dpsi - Uvals * psi
            phys = (r * r).mean()
            pic, dpic, _ = model.forward_with_derivs(self.aic, order=2)
            e0 = pic - Tensor(self.psi0)
            e1 = dpic - Tensor(self.dpsi0)
            bc = (e0 * e0).mean() + (e1 * e1).mean()
            return phys + Tensor(100.0) * bc
        return closure
