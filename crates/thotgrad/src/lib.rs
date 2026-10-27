//! thotgrad — Railway-style reverse-mode autodiff + optimisers, with PyO3.
//!
//! The pure numerical engine lives in [`core`] and is tested without Python
//! (`cargo test --no-default-features`). This file is the thin Python skin:
//! every `core::ThotError` crosses the boundary as a `ValueError`, so the
//! monadic error track in Rust becomes an ordinary Python exception.

pub mod core;

#[cfg(feature = "python")]
mod py {
    use crate::core;
    use ndarray::Array1;
    use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    impl From<core::ThotError> for PyErr {
        fn from(e: core::ThotError) -> PyErr {
            PyValueError::new_err(e.to_string())
        }
    }

    /// Tanh MLP with exact input-derivatives (drop-in for `thotgrad.MLP`).
    #[pyclass(name = "Mlp")]
    struct PyMlp {
        inner: core::Mlp,
    }

    #[pymethods]
    impl PyMlp {
        #[new]
        #[pyo3(signature = (n_in=1, n_out=1, width=32, depth=3, seed=0))]
        fn new(n_in: usize, n_out: usize, width: usize, depth: usize, seed: u64) -> Self {
            PyMlp { inner: core::Mlp::new(n_in, n_out, width, depth, seed) }
        }

        fn n_params(&self) -> usize {
            self.inner.n_params()
        }

        fn forward<'py>(
            &self,
            py: Python<'py>,
            x: PyReadonlyArray2<'py, f64>,
        ) -> PyResult<Bound<'py, PyArray2<f64>>> {
            let u = self.inner.forward(&x.as_array().to_owned())?;
            Ok(u.into_pyarray_bound(py))
        }

        /// `(u, u_x, u_xx)` for a column input `x[N,1]`.
        fn forward_with_derivs<'py>(
            &self,
            py: Python<'py>,
            x: PyReadonlyArray2<'py, f64>,
        ) -> PyResult<(
            Bound<'py, PyArray2<f64>>,
            Bound<'py, PyArray2<f64>>,
            Bound<'py, PyArray2<f64>>,
        )> {
            let (u, ux, uxx) = self.inner.forward_with_derivs(&x.as_array().to_owned())?;
            Ok((
                u.into_pyarray_bound(py),
                ux.into_pyarray_bound(py),
                uxx.into_pyarray_bound(py),
            ))
        }

        fn get_flat<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.inner.get_flat().into_pyarray_bound(py)
        }

        fn set_flat(&mut self, theta: PyReadonlyArray1<f64>) -> PyResult<()> {
            self.inner.set_flat(theta.as_slice()?)?;
            Ok(())
        }
    }

    /// The 1-D Poisson PINN benchmark problem, solved entirely in Rust.
    #[pyclass(name = "PoissonPinn")]
    struct PyPoisson {
        inner: core::PoissonPinn,
    }

    #[pymethods]
    impl PyPoisson {
        #[new]
        #[pyo3(signature = (width=32, depth=3, seed=0, n_coll=256))]
        fn new(width: usize, depth: usize, seed: u64, n_coll: usize) -> Self {
            PyPoisson { inner: core::PoissonPinn::new(width, depth, seed, n_coll) }
        }

        fn n_params(&self) -> usize {
            self.inner.mlp.n_params()
        }

        /// One forward+backward: `(loss, grad)` for a flat parameter vector.
        fn loss_and_grad<'py>(
            &mut self,
            py: Python<'py>,
            theta: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
            let (loss, grad) = self.inner.loss_and_grad(theta.as_slice()?)?;
            Ok((loss, grad.into_pyarray_bound(py)))
        }

        /// Full AdamW training loop in Rust: `(theta, loss_history)`.
        #[pyo3(signature = (steps=2000, lr=1e-3, weight_decay=0.0))]
        fn train_adamw<'py>(
            &mut self,
            py: Python<'py>,
            steps: usize,
            lr: f64,
            weight_decay: f64,
        ) -> PyResult<(Bound<'py, PyArray1<f64>>, Vec<f64>)> {
            let (theta, hist) = core::train_adamw(&mut self.inner, steps, lr, weight_decay)?;
            Ok((theta.into_pyarray_bound(py), hist))
        }

        /// Full L-BFGS training loop in Rust: `(theta, final_loss, n_iter)`.
        #[pyo3(signature = (max_iter=500, m=20, tol_grad=1e-12, c1=1e-4, max_ls=25))]
        fn train_lbfgs<'py>(
            &mut self,
            py: Python<'py>,
            max_iter: usize,
            m: usize,
            tol_grad: f64,
            c1: f64,
            max_ls: usize,
        ) -> PyResult<(Bound<'py, PyArray1<f64>>, f64, usize)> {
            let x0 = self.inner.mlp.get_flat();
            let (x, f, it) = core::lbfgs(&mut self.inner, x0, max_iter, m, tol_grad, c1, max_ls)?;
            Ok((x.into_pyarray_bound(py), f, it))
        }

        fn l2_error(&self) -> PyResult<f64> {
            Ok(self.inner.l2_error()?)
        }
    }

    /// AdamW optimiser (drop-in for `thotopt.AdamW`).
    #[pyclass(name = "AdamW")]
    struct PyAdamW {
        inner: core::AdamW,
    }

    #[pymethods]
    impl PyAdamW {
        #[new]
        #[pyo3(signature = (lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0))]
        fn new(lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
            PyAdamW { inner: core::AdamW::new(lr, beta1, beta2, eps, weight_decay) }
        }

        fn step<'py>(
            &mut self,
            py: Python<'py>,
            theta: PyReadonlyArray1<'py, f64>,
            grad: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let th: Array1<f64> = theta.as_array().to_owned();
            let g: Array1<f64> = grad.as_array().to_owned();
            Ok(self.inner.step(&th, &g).into_pyarray_bound(py))
        }
    }

    #[pymodule]
    fn thotgrad_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyMlp>()?;
        m.add_class::<PyPoisson>()?;
        m.add_class::<PyAdamW>()?;
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        Ok(())
    }
}
