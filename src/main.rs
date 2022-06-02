mod femsolver;
use femsolver::*;
use textplots::{Chart, Plot, Shape};

fn main() {
    let ab = (-1.0, 1.0);
    let uab = (0.0, 0.0);
    let mu = Box::new(|_: f64| 1.0);
    let beta = Box::new(|x: f64| 1925.0 * x * x);
    let beta2 = Box::new(|x: f64|80.0 * x);
    let sigma = Box::new(|x: f64| 500.0 * (x + 2.0).powi(3));
    let sigma2 = Box::new(|_: f64| 30.0);
    let f = Box::new(|x: f64| 17.0 * x / (49.0 * x * x + 1.0));
    let alpha = 1e+5f64;
    let gamma = 1e+5f64;
    let fem = HAdaptiveFemSolver {
        ab,
        uab,
        mu,
        beta : beta2,
        sigma : sigma2,
        f,
        alpha,
        gamma,
        startn: 5,
    };
    let res = fem.solve(0.1);
    let plotting : Vec<SolveResults> = res.iter().map(|f| f.0.clone()).collect();
    println!("\n----------------------------------------------------------------------\n");
    println!(
        "{:^17}  {:^17}  {:^17}  {:^17}  {:^17}  {:^17}  {:^17}  {:^17}", "Iteration", "Elements",
        "U Normalized", "U Normalized ^ 2", "E Normalized", "E Normalized ^ 2", "OrdOfConvergence", "Max Error"
    );
    println!(
        "{:^17}  {:^17}  {:^17}  {:^17}  {:^17}  {:^17}  {:^17}  {:^17}", "------", "------",
        "------", "------", "------", "------", "------", "------"
    );
    res.iter().enumerate().for_each(|(i, (r, e))| {
        println!("{:^17}  {:^17}  {}", i+1, r.finite_elements.len()+1, e.to_str());
    });
    println!("\n----------------------------------------------------------------------\n");
    println!("\n");
    plot_every(&plotting, 1);
}

fn plot_every(results: &[SolveResults], iteration: usize) {
    for i in (0..results.len()).step_by(iteration)
    {
        println!("\nIteration {}\n", i + 1);
        Chart::new(180, 60, -1.0, 1.0)
        .lineplot(&Shape::Lines(&results[i].for_plot()))
        .display();
    }
    if results.len() % iteration != 0 {
        println!("\nFinal Iteration ({})\n", results.len());
        Chart::new(180, 60, -1.0, 1.0)
        .lineplot(&Shape::Lines(&results[results.len() - 1].for_plot()))
        .display();
    }
}

