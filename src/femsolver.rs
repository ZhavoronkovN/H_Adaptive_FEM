use std::error::Error;

use itertools_num::linspace;
use ndarray::{self, Array1, Array2};
use ndarray_linalg::{self,Solve};

type MyResult<R> = Result<R, Box<dyn Error>>;

pub fn err<R>(message: &str) -> MyResult<R> {
    Result::Err(Box::new(std::io::Error::new(
        std::io::ErrorKind::Other,
        message,
    )))
}
#[derive(Clone)]
pub struct SolveErrors {
    errors: Vec<f64>,
    unorm: f64,
    unorm2: f64,
    enorm: f64,
    enorm2: f64,
    ofc: f64,
    max: f64,
}

impl SolveErrors {
    pub fn to_str(&self) -> String {
        format!(
            "{:^7.3}  {:^7.3}  {:^7.3}  {:^7.3}  {:^7.3}  {:^7.3}",
            self.unorm, self.unorm2, self.enorm, self.enorm2, self.ofc, self.max
        )
    }
    pub fn new(
        fem: &HAdaptiveFemSolver,
        results: &SolveResults,
        initial: Option<f64>,
    ) -> SolveErrors {
        println!("HERE");
        let mut norm_errors: Vec<f64> = results
            .all()
            .iter()
            .map(|(fe, _, cen, cend)| {
                let m = fe.h.powi(3) / (fem.mu)(fe.mid);
                let b = (fem.f)(fe.mid) - ((fem.beta)(fe.mid) * cend) - ((fem.sigma)(fe.mid) * cen);
                let d = 10.0
                    + ((fe.h * (fem.beta)(fe.mid)) / (fem.mu)(fe.mid)
                        * ((fe.h * fe.h * (fem.sigma)(fe.mid)) / (fem.mu)(fe.mid)));
                (5.0 / 6.0 * m * b * b / d).abs()
            })
            .collect();
        let enorm2: f64 = norm_errors.iter().sum();
        norm_errors.iter_mut().for_each(|f| *f = f.sqrt());
        let unorm2: f64 = results
            .all()
            .iter()
            .map(|(fe, _, _, cend)| fe.h * cend * cend)
            .sum();
        println!("Results size : {}", results.solutions.len());
        let m = fem.create_matrix_l(&results.finite_elements).0;
        println!("Matrix size : {:?}", m.dim());
        let unorm = Array1::from(results.solutions.clone())
            .dot(&m)
            .dot(&Array1::from(results.solutions.clone()))
            .sqrt();
        let enorm = enorm2.sqrt();
        let nsqrt = (results.solutions.len() as f64).sqrt();
        let errors: Vec<f64> = norm_errors
            .iter()
            .map(|f| (f * nsqrt * 100.0) / (unorm2 + enorm2).sqrt())
            .collect();
        let ofc = match initial {
            Some(v) => {
                (v.ln() - enorm.ln())
                    / ((results.finite_elements.len() as f64).ln() - (fem.startn as f64).ln())
            }
            None => 0.0,
        };
        let max = errors
            .iter()
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(&0.0)
            .clone();
        SolveErrors {
            errors,
            unorm,
            unorm2,
            enorm,
            enorm2,
            ofc,
            max,
        }
    }
}
#[derive(Clone)]
pub struct SolveResults {
    finite_elements: Vec<FiniteElement>,
    solutions: Vec<f64>,
    centers: Vec<f64>,
    centerd: Vec<f64>,
}

impl SolveResults {
    pub fn new(finite_elements: Vec<FiniteElement>, solutions: Vec<f64>) -> SolveResults {
        let centers = solutions
            .chunks_exact(2)
            .into_iter()
            .map(|c| (c[0] + c[1]) / 2.0)
            .collect();
        let centerd = solutions
            .chunks_exact(2)
            .into_iter()
            .zip(finite_elements[..finite_elements.len() - 1].into_iter())
            .map(|(c, fe)| (c[1] - c[0]) / fe.h)
            .collect();
        SolveResults {
            finite_elements,
            solutions,
            centers,
            centerd,
        }
    }

    pub fn all(&self) -> Vec<(FiniteElement, f64, f64, f64)> {
        self.finite_elements
            .iter()
            .zip(self.solutions.iter())
            .zip(self.centers.iter())
            .zip(self.centerd.iter())
            .map(|(((fe, sol), center), centerd)| {
                (fe.clone(), sol.clone(), center.clone(), centerd.clone())
            })
            .collect()
    }
}

#[derive(Clone)]
pub struct FiniteElement {
    start: f64,
    end: f64,
    mid: f64,
    h: f64,
}

impl FiniteElement {
    pub fn new(start: f64, end: f64) -> FiniteElement {
        return FiniteElement {
            start,
            end,
            mid: (end + start) / 2.0,
            h: (end - start) / 2.0,
        };
    }
}

pub struct HAdaptiveFemSolver {
    pub mu: Box<dyn Fn(f64) -> f64>,
    pub beta: Box<dyn Fn(f64) -> f64>,
    pub sigma: Box<dyn Fn(f64) -> f64>,
    pub f: Box<dyn Fn(f64) -> f64>,
    pub ab: (f64, f64),
    pub alpha: f64,
    pub gamma: f64,
    pub uab: (f64, f64),
    pub startn: usize,
}

impl HAdaptiveFemSolver {
    pub fn first_it(&self) -> (SolveResults, SolveErrors) {
        let els = self.init_finite_elements();
        let (m, l) = self.create_matrix_l(&els);
        let sol = HAdaptiveFemSolver::lin_eq_solve(m, l).unwrap().to_vec();
        let results = SolveResults::new(els.clone(), sol.clone());
        let errors = SolveErrors::new(self, &results, None);
        (results, errors)
    }
    pub fn init_finite_elements(&self) -> Vec<FiniteElement> {
        linspace::<f64>(self.ab.0, self.ab.1, self.startn)
            .into_iter()
            .zip(
                linspace::<f64>(self.ab.0, self.ab.1, self.startn)
                    .into_iter()
                    .skip(1),
            )
            .map(|(i, i1)| FiniteElement::new(i, i1))
            .collect()
    }

    pub fn create_matrix_l(&self, elements: &[FiniteElement]) -> (Array2<f64>, Array1<f64>) {
        let mut l = Vec::with_capacity(elements.len());
        let mut m = Array2::<f64>::zeros((elements.len() + 1, elements.len() + 1));
        let first = &elements[0];
        l.push(0.5 * first.h * (self.f)(first.mid) + self.alpha * self.uab.0);
        m[[0, 0]] = (self.mu)(first.mid) / first.h - (self.beta)(first.mid) / 2.0
            + (self.sigma)(first.mid) * first.h / 3.0
            + self.alpha;
        m[[0, 1]] = -(self.mu)(first.mid) / first.h
            + (self.beta)(first.mid) / 2.0
            + (self.sigma)(first.mid) * first.h / 6.0;
        elements[1..].iter().enumerate().for_each(|(i, fe)| {
            let prev = &elements[i - 1];
            l.push(0.5 * (prev.h * (self.f)(prev.mid)) + fe.h * (self.f)(fe.mid));
            m[[i, i - 1]] = -(self.mu)(prev.mid) / prev.h - (self.beta)(prev.mid) / 2.0
                + (self.sigma)(prev.mid) * prev.h / 6.0;
            m[[i, i]] = -(self.mu)(prev.mid) / prev.h - (self.beta)(prev.mid) / 2.0
                + (self.sigma)(prev.mid) * prev.h / 3.0
                + (self.mu)(fe.mid) / fe.h
                - (self.beta)(fe.mid) / 2.0
                + (self.sigma)(fe.mid) * fe.h / 3.0;
            m[[i, i + 1]] = -(self.mu)(fe.mid) / fe.h - (self.beta)(fe.mid) / 2.0
                + (self.sigma)(fe.mid) * fe.h / 6.0;
        });
        let last = &elements[elements.len() - 1];
        l.push(0.5 * last.h * (self.f)(last.mid) + self.gamma + self.uab.1);
        m[[elements.len(), elements.len() - 1]] = -(self.mu)(last.mid) / last.h
            + (self.beta)(last.mid) / 2.0
            + (self.sigma)(last.mid) * last.h / 6.0;
        m[[elements.len(), elements.len()]] = (self.mu)(last.mid) / last.h
            - (self.beta)(last.mid) / 2.0
            + (self.sigma)(first.mid) * last.h / 3.0
            + self.gamma;
        (m, Array1::from(l))
    }

    pub fn lin_eq_solve(matrix: Array2<f64>, l: Array1<f64>) -> MyResult<Array1<f64>> {
        matrix.solve_into(l).or(err("Failed to solve lin eq"))
    }

    pub fn solve(&self, error: f64) -> Vec<(SolveResults, SolveErrors)> {
        let (fres, ferr) = self.first_it();
        let initial_err = ferr.enorm.clone();
        let mut last_els = fres.finite_elements.clone();
        let mut last_ers = ferr.errors.clone();
        let mut total_results = vec![(fres.clone(), ferr.clone())];
        loop {
            let can_be_adapted: Vec<usize> = last_ers
                .iter()
                .enumerate()
                .filter(|(_, el)| **el > error)
                .map(|f| f.0.clone())
                .collect();
            if can_be_adapted.len() == 0 {
                break;
            }
            let mut new_elements = Vec::new();
            for el in 0..last_els.len() {
                match can_be_adapted.contains(&el) {
                    false => new_elements.push(last_els[el].clone()),
                    true => {
                        new_elements.push(FiniteElement::new(last_els[el].start, last_els[el].mid));
                        new_elements.push(FiniteElement::new(last_els[el].mid, last_els[el].end));
                    }
                }
            }
            let (m, l) = self.create_matrix_l(&new_elements);
            let sol = HAdaptiveFemSolver::lin_eq_solve(m, l).unwrap().to_vec();
            let results = SolveResults::new(new_elements.clone(), sol.clone());
            let errors = SolveErrors::new(self, &results, Some(initial_err));
            last_els = new_elements;
            last_ers = errors.errors.clone();
            total_results.push((results, errors));
        }
        total_results
    }
}
