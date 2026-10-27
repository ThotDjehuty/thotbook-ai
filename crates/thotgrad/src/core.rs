//! Railway-style reverse-mode automatic differentiation and optimisers.
//!
//! This is the Rust rewrite of `notebooks/thotgrad.py` + `notebooks/thotopt.py`.
//! The design keeps the Python algorithm exactly (array-level reverse-mode over
//! a tape; the analytic input-derivative recursion for a tanh MLP) but runs it
//! as native `ndarray` code with no interpreter dispatch per operation.
//!
//! "Railway style" (after the Polarway engine): every fallible step returns
//! [`Result<T, ThotError>`] and the pipelines compose with `?` / `and_then`,
//! so a non-finite value or a shape mismatch short-circuits the whole
//! expression down the error track instead of panicking. Nothing in the hot
//! path unwraps.

use ndarray::{Array1, Array2, Axis};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use thiserror::Error;

/// The single error track for every thotgrad pipeline.
#[derive(Debug, Error)]
pub enum ThotError {
    #[error("shape mismatch: {0}")]
    Shape(String),
    #[error("non-finite value in {0}")]
    NonFinite(&'static str),
    #[error("forward_with_derivs requires a single input feature (n_in == 1), got {0}")]
    NotScalarInput(usize),
    #[error("parameter vector has {got} entries, expected {want}")]
    BadTheta { got: usize, want: usize },
    #[error("line search failed to find a descent step")]
    LineSearch,
}

pub type Result<T> = std::result::Result<T, ThotError>;

#[inline]
fn finite(a: &Array2<f64>, where_: &'static str) -> Result<()> {
    if a.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(ThotError::NonFinite(where_))
    }
}

/// `a[N,m] + b[1,m]` with an explicit row broadcast.
fn add_bias(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
    let (n, m) = (a.nrows(), a.ncols());
    if b.nrows() != 1 || b.ncols() != m {
        return Err(ThotError::Shape(format!(
            "bias {:?} not broadcastable onto {:?}",
            b.dim(),
            a.dim()
        )));
    }
    let bc = b.broadcast((n, m)).unwrap();
    Ok(a + &bc)
}

// ---------------------------------------------------------------------------
// MLP with analytic input-derivatives (the PINN forward pass)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Layer {
    w: Array2<f64>, // [fan_in, fan_out]
    b: Array2<f64>, // [1, fan_out]
}

/// Fully-connected tanh network with exact first/second input-derivatives.
#[derive(Clone)]
pub struct Mlp {
    layers: Vec<Layer>,
    n_in: usize,
    n_out: usize,
}

impl Mlp {
    /// Xavier/Glorot-normal init, matching the Python `MLP` seed-for-seed is
    /// not attempted (different RNG), but the distribution is identical.
    pub fn new(n_in: usize, n_out: usize, width: usize, depth: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut dims = vec![n_in];
        dims.extend(std::iter::repeat(width).take(depth));
        dims.push(n_out);
        let mut layers = Vec::with_capacity(dims.len() - 1);
        for i in 0..dims.len() - 1 {
            let (fan_in, fan_out) = (dims[i], dims[i + 1]);
            let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
            let normal = Normal::new(0.0, std).unwrap();
            let w = Array2::from_shape_fn((fan_in, fan_out), |_| normal.sample(&mut rng));
            let b = Array2::zeros((1, fan_out));
            layers.push(Layer { w, b });
        }
        Mlp { layers, n_in, n_out }
    }

    pub fn n_params(&self) -> usize {
        self.layers.iter().map(|l| l.w.len() + l.b.len()).sum()
    }

    pub fn n_in(&self) -> usize {
        self.n_in
    }
    pub fn n_out(&self) -> usize {
        self.n_out
    }

    /// Plain forward pass, `x[N, n_in] -> u[N, n_out]`.
    pub fn forward(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let mut h = x.clone();
        let last = self.layers.len() - 1;
        for (i, l) in self.layers.iter().enumerate() {
            h = add_bias(&h.dot(&l.w), &l.b)?;
            if i < last {
                h.mapv_inplace(f64::tanh);
            }
        }
        finite(&h, "forward")?;
        Ok(h)
    }

    /// `(u, u_x, u_xx)` for a column input `x[N,1]`, propagated analytically.
    ///
    /// With `h = tanh(a)`, `t = 1 - h^2`:
    /// `h' = t * a'`, `h'' = t * (a'' - 2 h (a')^2)`.
    pub fn forward_with_derivs(
        &self,
        x: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>)> {
        if self.n_in != 1 {
            return Err(ThotError::NotScalarInput(self.n_in));
        }
        if x.ncols() != 1 {
            return Err(ThotError::Shape(format!("x must be [N,1], got {:?}", x.dim())));
        }
        let mut h = x.clone();
        let mut dh = Array2::<f64>::ones(x.dim());
        let mut d2h = Array2::<f64>::zeros(x.dim());
        let last = self.layers.len() - 1;
        for (i, l) in self.layers.iter().enumerate() {
            let a = add_bias(&h.dot(&l.w), &l.b)?;
            let da = dh.dot(&l.w);
            let d2a = d2h.dot(&l.w);
            if i < last {
                let hh = a.mapv(f64::tanh);
                let t = 1.0 - &hh * &hh; // sech^2
                dh = &t * &da;
                d2h = &t * &(&d2a - &(2.0 * &hh * &da * &da));
                h = hh;
            } else {
                h = a;
                dh = da;
                d2h = d2a;
            }
        }
        finite(&d2h, "forward_with_derivs")?;
        Ok((h, dh, d2h))
    }

    pub fn get_flat(&self) -> Array1<f64> {
        let mut v = Vec::with_capacity(self.n_params());
        for l in &self.layers {
            v.extend(l.w.iter().copied());
            v.extend(l.b.iter().copied());
        }
        Array1::from(v)
    }

    pub fn set_flat(&mut self, theta: &[f64]) -> Result<()> {
        let want = self.n_params();
        if theta.len() != want {
            return Err(ThotError::BadTheta { got: theta.len(), want });
        }
        let mut i = 0;
        for l in &mut self.layers {
            let nw = l.w.len();
            l.w.iter_mut().zip(&theta[i..i + nw]).for_each(|(d, &s)| *d = s);
            i += nw;
            let nb = l.b.len();
            l.b.iter_mut().zip(&theta[i..i + nb]).for_each(|(d, &s)| *d = s);
            i += nb;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Reverse-mode tape (index-based) — only the ops the PINN loss needs
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Op {
    Leaf,
    AddBias { a: usize, b: usize },
    MatMul { a: usize, b: usize },
    Mul { a: usize, b: usize },
    Add { a: usize, b: usize },
    Scale { a: usize, c: f64 },
    AddConst { a: usize },
    Tanh { a: usize },
    Mean { a: usize },
}

/// A throwaway tape built once per gradient evaluation. Values and gradients
/// live in separate vectors (struct-of-arrays): the reverse sweep reads operand
/// *values* and writes operand *gradients* without cloning either (only the one
/// upstream gradient is cloned per node), and a gradient buffer is allocated
/// only for nodes that actually require one -- constants carry a `0x0` stub.
pub struct Tape {
    vals: Vec<Array2<f64>>,
    grads: Vec<Array2<f64>>,
    reqs: Vec<bool>,
    ops: Vec<Op>,
}

impl Tape {
    fn new() -> Self {
        Tape { vals: Vec::new(), grads: Vec::new(), reqs: Vec::new(), ops: Vec::new() }
    }

    fn push(&mut self, val: Array2<f64>, req: bool, op: Op) -> usize {
        let grad = if req { Array2::zeros(val.dim()) } else { Array2::zeros((0, 0)) };
        self.vals.push(val);
        self.grads.push(grad);
        self.reqs.push(req);
        self.ops.push(op);
        self.vals.len() - 1
    }

    fn leaf(&mut self, val: Array2<f64>, req: bool) -> usize {
        self.push(val, req, Op::Leaf)
    }

    fn matmul(&mut self, a: usize, b: usize) -> usize {
        let val = self.vals[a].dot(&self.vals[b]);
        self.push(val, self.reqs[a] || self.reqs[b], Op::MatMul { a, b })
    }

    fn add_bias(&mut self, a: usize, b: usize) -> Result<usize> {
        let val = add_bias(&self.vals[a], &self.vals[b])?;
        Ok(self.push(val, self.reqs[a] || self.reqs[b], Op::AddBias { a, b }))
    }

    fn mul(&mut self, a: usize, b: usize) -> usize {
        let val = &self.vals[a] * &self.vals[b];
        self.push(val, self.reqs[a] || self.reqs[b], Op::Mul { a, b })
    }

    fn add(&mut self, a: usize, b: usize) -> usize {
        let val = &self.vals[a] + &self.vals[b];
        self.push(val, self.reqs[a] || self.reqs[b], Op::Add { a, b })
    }

    fn scale(&mut self, a: usize, c: f64) -> usize {
        let val = &self.vals[a] * c;
        self.push(val, self.reqs[a], Op::Scale { a, c })
    }

    fn add_const(&mut self, a: usize, c: f64) -> usize {
        let val = &self.vals[a] + c;
        self.push(val, self.reqs[a], Op::AddConst { a })
    }

    fn tanh(&mut self, a: usize) -> usize {
        let val = self.vals[a].mapv(f64::tanh);
        self.push(val, self.reqs[a], Op::Tanh { a })
    }

    fn mean(&mut self, a: usize) -> usize {
        let m = self.vals[a].mean().unwrap_or(0.0);
        self.push(Array2::from_elem((1, 1), m), self.reqs[a], Op::Mean { a })
    }

    fn value(&self, i: usize) -> f64 {
        self.vals[i][[0, 0]]
    }

    /// Reverse sweep seeded at the scalar node `root`.
    fn backward(&mut self, root: usize) {
        for i in 0..self.grads.len() {
            if self.reqs[i] {
                let d = self.vals[i].dim();
                self.grads[i] = Array2::zeros(d);
            }
        }
        self.grads[root][[0, 0]] = 1.0;
        for i in (0..self.ops.len()).rev() {
            if !self.reqs[i] {
                continue;
            }
            let op = self.ops[i];
            let g = self.grads[i].clone();
            match op {
                Op::Leaf => {}
                Op::AddBias { a, b } => {
                    if self.reqs[a] {
                        self.grads[a] += &g;
                    }
                    if self.reqs[b] {
                        let col = g.sum_axis(Axis(0)).insert_axis(Axis(0));
                        self.grads[b] += &col;
                    }
                }
                Op::MatMul { a, b } => {
                    if self.reqs[a] {
                        let t = g.dot(&self.vals[b].t());
                        self.grads[a] += &t;
                    }
                    if self.reqs[b] {
                        let t = self.vals[a].t().dot(&g);
                        self.grads[b] += &t;
                    }
                }
                Op::Mul { a, b } => {
                    if self.reqs[a] {
                        let t = &g * &self.vals[b];
                        self.grads[a] += &t;
                    }
                    if self.reqs[b] {
                        let t = &g * &self.vals[a];
                        self.grads[b] += &t;
                    }
                }
                Op::Add { a, b } => {
                    if self.reqs[a] {
                        self.grads[a] += &g;
                    }
                    if self.reqs[b] {
                        self.grads[b] += &g;
                    }
                }
                Op::Scale { a, c } => {
                    if self.reqs[a] {
                        let t = &g * c;
                        self.grads[a] += &t;
                    }
                }
                Op::AddConst { a } => {
                    if self.reqs[a] {
                        self.grads[a] += &g;
                    }
                }
                Op::Tanh { a } => {
                    if self.reqs[a] {
                        let v = &self.vals[i];
                        let d = 1.0 - v * v;
                        let t = &g * &d;
                        self.grads[a] += &t;
                    }
                }
                Op::Mean { a } => {
                    if self.reqs[a] {
                        let dim = self.vals[a].dim();
                        let nn = self.vals[a].len() as f64;
                        let gv = g[[0, 0]] / nn;
                        self.grads[a] += &Array2::from_elem(dim, gv);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 1-D Poisson PINN — the benchmark problem
//   u''(x) = f(x) = -pi^2 sin(pi x)  on [0,1],  u(0)=u(1)=0
//   exact: u(x) = sin(pi x)
// ---------------------------------------------------------------------------

pub struct PoissonPinn {
    pub mlp: Mlp,
    xc: Array2<f64>,   // collocation points [N,1]
    f: Array2<f64>,    // rhs at collocation [N,1]
    xb: Array2<f64>,   // boundary points [2,1]
    ub: Array2<f64>,   // boundary values [2,1]
    w_bc: f64,
}

impl PoissonPinn {
    pub fn new(width: usize, depth: usize, seed: u64, n_coll: usize) -> Self {
        let mlp = Mlp::new(1, 1, width, depth, seed);
        let xc = Array2::from_shape_fn((n_coll, 1), |(i, _)| {
            (i as f64 + 1.0) / (n_coll as f64 + 1.0)
        });
        let pi = std::f64::consts::PI;
        let f = xc.mapv(|x| -(pi * pi) * (pi * x).sin());
        let xb = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let ub = Array2::zeros((2, 1));
        PoissonPinn { mlp, xc, f, xb, ub, w_bc: 1.0 }
    }

    /// Build the tape for the PINN loss and return `(loss, d loss / d theta)`.
    pub fn loss_and_grad(&mut self, theta: &[f64]) -> Result<(f64, Array1<f64>)> {
        self.mlp.set_flat(theta)?;
        let mut tp = Tape::new();

        // leaves: parameters (require grad)
        let mut wl = Vec::new();
        let mut bl = Vec::new();
        for l in &self.mlp.layers {
            wl.push(tp.leaf(l.w.clone(), true));
            bl.push(tp.leaf(l.b.clone(), true));
        }

        // ---- interior: u_xx via the analytic deriv recursion, on the tape ----
        let xc = tp.leaf(self.xc.clone(), false);
        let one = tp.leaf(Array2::ones(self.xc.dim()), false);
        let zero = tp.leaf(Array2::zeros(self.xc.dim()), false);
        let (mut h, mut dh, mut d2h) = (xc, one, zero);
        let last = self.mlp.layers.len() - 1;
        for i in 0..self.mlp.layers.len() {
            let a = {
                let m = tp.matmul(h, wl[i]);
                tp.add_bias(m, bl[i])?
            };
            let da = tp.matmul(dh, wl[i]);
            let d2a = tp.matmul(d2h, wl[i]);
            if i < last {
                let hh = tp.tanh(a);
                let hsq = tp.mul(hh, hh);
                let neg_hsq = tp.scale(hsq, -1.0);
                let t = tp.add_const(neg_hsq, 1.0); // 1 - h^2
                dh = tp.mul(t, da);
                // d2h = t * (d2a - 2 h (da)^2)
                let da2 = tp.mul(da, da);
                let hda2 = tp.mul(hh, da2);
                let neg2_hda2 = tp.scale(hda2, -2.0);
                let inner = tp.add(d2a, neg2_hda2);
                d2h = tp.mul(t, inner);
                h = hh;
            } else {
                h = a;
                dh = da;
                d2h = d2a;
            }
        }
        let _ = (h, dh); // u and u_x unused by the residual, keep for clarity

        // residual r = u_xx - f ; loss_res = mean(r^2)
        let fnode = tp.leaf(self.f.clone(), false);
        let negf = tp.scale(fnode, -1.0);
        let r = tp.add(d2h, negf);
        let r2 = tp.mul(r, r);
        let loss_res = tp.mean(r2);

        // ---- boundary: plain forward at xb, loss_bc = mean((u_bc - ub)^2) ----
        let xb = tp.leaf(self.xb.clone(), false);
        let mut hb = xb;
        for i in 0..self.mlp.layers.len() {
            let m = tp.matmul(hb, wl[i]);
            let ab = tp.add_bias(m, bl[i])?;
            hb = if i < last { tp.tanh(ab) } else { ab };
        }
        let ubnode = tp.leaf(self.ub.clone(), false);
        let negub = tp.scale(ubnode, -1.0);
        let rb = tp.add(hb, negub);
        let rb2 = tp.mul(rb, rb);
        let loss_bc = tp.mean(rb2);

        // total = loss_res + w_bc * loss_bc
        let w_loss_bc = tp.scale(loss_bc, self.w_bc);
        let total = tp.add(loss_res, w_loss_bc);
        let loss = tp.value(total);
        if !loss.is_finite() {
            return Err(ThotError::NonFinite("loss"));
        }

        tp.backward(total);

        // gather parameter grads in get_flat order
        let mut grad = Vec::with_capacity(self.mlp.n_params());
        for i in 0..self.mlp.layers.len() {
            grad.extend(tp.grads[wl[i]].iter().copied());
            grad.extend(tp.grads[bl[i]].iter().copied());
        }
        Ok((loss, Array1::from(grad)))
    }

    /// L2 error of the current network against the exact solution sin(pi x).
    pub fn l2_error(&self) -> Result<f64> {
        let pi = std::f64::consts::PI;
        let u = self.mlp.forward(&self.xc)?;
        let exact = self.xc.mapv(|x| (pi * x).sin());
        let num: f64 = u
            .iter()
            .zip(exact.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let den: f64 = exact.iter().map(|b| b * b).sum();
        Ok((num / den).sqrt())
    }
}

// ---------------------------------------------------------------------------
// Optimisers (thotopt)
// ---------------------------------------------------------------------------

/// Adam with decoupled weight decay (Loshchilov & Hutter 2019).
pub struct AdamW {
    lr: f64,
    b1: f64,
    b2: f64,
    eps: f64,
    wd: f64,
    m: Option<Array1<f64>>,
    v: Option<Array1<f64>>,
    t: i32,
}

impl AdamW {
    pub fn new(lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        AdamW { lr, b1: beta1, b2: beta2, eps, wd: weight_decay, m: None, v: None, t: 0 }
    }

    pub fn step(&mut self, theta: &Array1<f64>, grad: &Array1<f64>) -> Array1<f64> {
        if self.m.is_none() {
            self.m = Some(Array1::zeros(theta.len()));
            self.v = Some(Array1::zeros(theta.len()));
        }
        self.t += 1;
        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();
        *m = self.b1 * &*m + (1.0 - self.b1) * grad;
        *v = self.b2 * &*v + (1.0 - self.b2) * (grad * grad);
        let mc = 1.0 - self.b1.powi(self.t);
        let vc = 1.0 - self.b2.powi(self.t);
        let mut out = theta.clone();
        ndarray::Zip::from(&mut out)
            .and(&*m)
            .and(&*v)
            .and(theta)
            .for_each(|o, &mi, &vi, &th| {
                let mhat = mi / mc;
                let vhat = vi / vc;
                *o = th - self.lr * (mhat / (vhat.sqrt() + self.eps) + self.wd * th);
            });
        out
    }
}

/// Train the Poisson PINN with AdamW entirely in Rust.
pub fn train_adamw(
    pinn: &mut PoissonPinn,
    steps: usize,
    lr: f64,
    weight_decay: f64,
) -> Result<(Array1<f64>, Vec<f64>)> {
    let mut theta = pinn.mlp.get_flat();
    let mut opt = AdamW::new(lr, 0.9, 0.999, 1e-8, weight_decay);
    let mut hist = Vec::with_capacity(steps);
    for _ in 0..steps {
        let (loss, grad) = pinn.loss_and_grad(theta.as_slice().unwrap())?;
        hist.push(loss);
        theta = opt.step(&theta, &grad);
    }
    pinn.mlp.set_flat(theta.as_slice().unwrap())?;
    Ok((theta, hist))
}

/// Limited-memory BFGS (Nocedal two-loop) + backtracking Armijo, in Rust.
pub fn lbfgs(
    pinn: &mut PoissonPinn,
    x0: Array1<f64>,
    max_iter: usize,
    m: usize,
    tol_grad: f64,
    c1: f64,
    max_ls: usize,
) -> Result<(Array1<f64>, f64, usize)> {
    let dot = |a: &Array1<f64>, b: &Array1<f64>| a.dot(b);
    let mut x = x0;
    let (mut f, mut g) = pinn.loss_and_grad(x.as_slice().unwrap())?;
    let mut s_hist: std::collections::VecDeque<Array1<f64>> = std::collections::VecDeque::new();
    let mut y_hist: std::collections::VecDeque<Array1<f64>> = std::collections::VecDeque::new();
    let mut it = 0;

    for k in 1..=max_iter {
        it = k;
        let gnorm = dot(&g, &g).sqrt();
        if gnorm < tol_grad || !f.is_finite() {
            break;
        }

        // two-loop recursion: q <- H_k g
        let mut q = g.clone();
        let mut alphas: Vec<(f64, f64)> = Vec::with_capacity(s_hist.len());
        for (s_i, y_i) in s_hist.iter().rev().zip(y_hist.iter().rev()) {
            let rho = 1.0 / dot(y_i, s_i);
            let a = rho * dot(s_i, &q);
            alphas.push((rho, a));
            q = &q - &(a * y_i);
        }
        if let (Some(s_last), Some(y_last)) = (s_hist.back(), y_hist.back()) {
            q *= dot(s_last, y_last) / dot(y_last, y_last);
        }
        for ((rho, a), (s_i, y_i)) in alphas
            .iter()
            .rev()
            .zip(s_hist.iter().zip(y_hist.iter()))
        {
            let beta = rho * dot(y_i, &q);
            q = &q + &((a - beta) * s_i);
        }
        let mut p = -&q;
        if dot(&p, &g) >= 0.0 {
            p = -&g;
            s_hist.clear();
            y_hist.clear();
        }

        // backtracking Armijo
        let gp = dot(&p, &g);
        let mut step = 1.0;
        let mut ok = false;
        let (mut f_new, mut g_new, mut x_new) = (f, g.clone(), x.clone());
        for _ in 0..max_ls {
            x_new = &x + &(step * &p);
            let (fn_, gn_) = pinn.loss_and_grad(x_new.as_slice().unwrap())?;
            f_new = fn_;
            g_new = gn_;
            if f_new.is_finite() && f_new <= f + c1 * step * gp {
                ok = true;
                break;
            }
            step *= 0.5;
        }
        if !ok {
            break;
        }

        let s = &x_new - &x;
        let y = &g_new - &g;
        if dot(&s, &y) > 1e-12 {
            if s_hist.len() == m {
                s_hist.pop_front();
                y_hist.pop_front();
            }
            s_hist.push_back(s);
            y_hist.push_back(y);
        }
        x = x_new;
        f = f_new;
        g = g_new;
    }
    pinn.mlp.set_flat(x.as_slice().unwrap())?;
    Ok((x, f, it))
}

// ---------------------------------------------------------------------------
// Tests — pinned against central finite differences, like test_thotgrad.py
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_derivs_match_fd() {
        let mlp = Mlp::new(1, 1, 16, 3, 7);
        let x = Array2::from_shape_fn((5, 1), |(i, _)| 0.1 + 0.15 * i as f64);
        let (_, ux, uxx) = mlp.forward_with_derivs(&x).unwrap();
        let h = 1e-6;
        let xp = &x + h;
        let xm = &x - h;
        let up = mlp.forward(&xp).unwrap();
        let um = mlp.forward(&xm).unwrap();
        let u0 = mlp.forward(&x).unwrap();
        let fd1 = (&up - &um) / (2.0 * h);
        let fd2 = (&up - &(2.0 * &u0) + &um) / (h * h);
        let e1 = (&ux - &fd1).iter().map(|v| v.abs()).fold(0.0, f64::max);
        let e2 = (&uxx - &fd2).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(e1 < 1e-5, "u_x vs FD max abs err {e1}");
        assert!(e2 < 1e-2, "u_xx vs FD max abs err {e2}");
    }

    #[test]
    fn param_grad_matches_fd() {
        let mut pinn = PoissonPinn::new(12, 2, 3, 24);
        let theta = pinn.mlp.get_flat();
        let (_, grad) = pinn.loss_and_grad(theta.as_slice().unwrap()).unwrap();
        let h = 1e-6;
        let mut max_err = 0.0_f64;
        // check a handful of coordinates (full FD is O(P) forward passes)
        for &j in &[0usize, 5, 11, 17, 23, theta.len() - 1] {
            let mut tp = theta.clone();
            let mut tm = theta.clone();
            tp[j] += h;
            tm[j] -= h;
            let (fp, _) = pinn.loss_and_grad(tp.as_slice().unwrap()).unwrap();
            let (fm, _) = pinn.loss_and_grad(tm.as_slice().unwrap()).unwrap();
            let fd = (fp - fm) / (2.0 * h);
            max_err = max_err.max((grad[j] - fd).abs());
        }
        assert!(max_err < 1e-5, "param grad vs FD max abs err {max_err}");
    }

    #[test]
    fn adamw_decays_at_zero_grad() {
        // with zero gradient, AdamW should shrink theta by exactly lr*wd*theta
        let mut opt = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.5);
        let theta = Array1::from(vec![1.0, -2.0, 3.0]);
        let zero = Array1::zeros(3);
        let out = opt.step(&theta, &zero);
        for (o, th) in out.iter().zip(theta.iter()) {
            assert!((o - (th - 0.1 * 0.5 * th)).abs() < 1e-12);
        }
    }

    #[test]
    fn lbfgs_reaches_low_loss() {
        let mut pinn = PoissonPinn::new(24, 3, 0, 64);
        let x0 = pinn.mlp.get_flat();
        let (_, f, _) = lbfgs(&mut pinn, x0, 200, 20, 1e-12, 1e-4, 25).unwrap();
        assert!(f < 1e-2, "L-BFGS final loss {f} not < 1e-2");
    }
}
