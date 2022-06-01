mod femsolver;
use femsolver::*;

fn main() {
    let ab = (-1.0, 1.0);
    let uab = (0.0,0.0);
    let mu = Box::new(|_ : f64| 1.0);
    let beta = Box::new(|x : f64| 1500.0 * x.powi(8));
    let sigma = Box::new(|x : f64| 80.0 + 2.0 * x.powi(2));
    let f = Box::new(|x : f64| 100.0 * x * f64::exp((x-0.15).powi(7)));
    let alpha = 100000f64;
    let gamma = 100000f64;
    let fem = HAdaptiveFemSolver{ab,uab,mu,beta,sigma,f,alpha,gamma,startn: 20};
    let res = fem.solve(1.0);
    res.iter().enumerate().for_each(|(i,(_,e))| {
        println!("{} -- {}",i,e.to_str());
    });
    println!("Hello, world!");
}
