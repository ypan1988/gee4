#include <RcppArmadillo.h>
#include <string>

#include "newton.h"
#include "gee_jmcm.h"

// [[Rcpp::export]]
Rcpp::List gees_estimation(arma::uvec m, arma::vec Y, arma::mat X, arma::mat Z, arma::mat W,
			   std::string corrStruct, double rho, arma::vec start,
			   bool trace = false, bool profile = true, bool errorMsg = false)
{
  int debug = 1;

  if (debug) Rcpp::Rcout << "gees_estimation(): " << std::endl;
  
  int n_bta = X.n_cols;
  int n_lmd = Z.n_cols;
  int n_gma = W.n_cols;

  if (debug) Rcpp::Rcout << "gees_estimation(): setting corr_mode..." << std::endl;
  gee_corr_mode corr_mode(0);
  if (corrStruct == "id") corr_mode.setid(1);
  else if (corrStruct == "cs") corr_mode.setid(2);
  else if (corrStruct == "ar1") corr_mode.setid(3);

  if (debug) Rcpp::Rcout << "gees_estimation(): creating gees object..." << std::endl;
  gee::gee_jmcm gees(m, Y, X, Z, W, rho, identity_link, corr_mode);
  dragonwell::Newton<gee::gee_jmcm> newt(gees);
  dragonwell::LineSearch<dragonwell::NRfmin<gee::gee_jmcm>> linesearch;
  linesearch.set_message(errorMsg);
  
  arma::vec x = start;

  //double f_min = 0.0;
  int n_iters  = 0;

////////////////////////////////////////////////////
  if (profile) {
    if(debug) {
      Rcpp::Rcout << "gees_estimation(): Start profile optimization ... " << std::endl;
      x.t().print("start value: ");
    }

    bool check;
    const arma::uword kMaxIters = 2000; // maximum number of iterations
    const double kTolF = 1.0e-8;	// convergence criterion on the function value
    const double kTolMin = 1.0e-12;	// criterion for deciding whether spurious
                                        // convergence to a minimum of fmin has occurred 
    const double kStpMax = 100.0;	// scaled maximum step length allowed in line searches
    const double kTolX = std::numeric_limits<double>::epsilon(); // convergence criterion on x

    const arma::uword n = x.n_elem;
    dragonwell::NRfmin<gee::gee_jmcm> fmin(gees);
    dragonwell::NRfdjac<gee::gee_jmcm> fdjac(gees);
    arma::vec &fvec = fmin.fvec;
    double f = fmin(x);

    double test = 0.0;
    for (arma::uword i = 0; i < n; ++i) {
      if (std::abs(fvec(i)) > test) test = std::abs(fvec(i));
    }
    if (test < 0.01 * kTolF) {
      check = false;
      // return check;
    }
    
    double sum = arma::as_scalar(x.t() * x);
    double stpmax = kStpMax * std::max(std::sqrt(sum), (double)n);

    arma::mat fjac = arma::zeros<arma::mat>(n, n);
    for (arma::uword iter = 0; iter < kMaxIters; ++iter) {
      fjac = fdjac(x, fvec);
      arma::vec g = fjac.t() * fvec;
      arma::vec xold = x;
      
      if (debug) Rcpp::Rcout << "Update beta..." << std::endl;
      gees.UpdateBeta();
      if (debug) Rcpp::Rcout << "Update lambda..." << std::endl;
      gees.UpdateLambda();
      if (debug) Rcpp::Rcout << "Update gamma..." << std::endl;
      gees.UpdateGamma();
      if (debug) Rcpp::Rcout << "Update theta..." << std::endl;
      arma::vec xnew = gees.get_theta();
      if (debug) xnew.t().print("xnew = ");
      arma::vec p = xnew - x;
      
      x = xnew;
      // fjac = fdjac(x, fvec);
      // arma::vec g = fjac.t() * fvec;
      // arma::vec xold = x;
      // // fold = f;
      // arma::vec p = -fvec;

      // //p = arma::solve(fjac, p);
      // arma::vec ptmp;
      // if (arma::solve(ptmp, fjac, p)) p = ptmp;
      // else p = arma::pinv(fjac.t() * fjac) * fjac.t() * p;
      // check = linesearch.GetStep(fmin, &f, &x, g, p, stpmax);

      if (trace) {
        Rcpp::Rcout << iter << ": " << f << ": " << x.t() << std::endl;
      }

      test = 0.0;
      for (arma::uword i = 0; i < n; ++i) {
        if (std::abs(fvec(i)) > test) test = std::abs(fvec(i));
      }

      if (test < 0.01 * kTolF) {
        check = false;
	break;  // return check;
      }
      if (check) {
        test = 0.0;
        double den = std::max(f, 0.5*n);
        for(arma::uword i = 0; i < n; ++i) {
          double temp = std::abs(g(i)) * std::max(std::abs(x(i)),1.0)/den;
          if(temp > test) test = temp;
        }
        check = (test < kTolMin);
        break;  // return check;
      }

      test = 0.0;
      for (arma::uword i = 0; i < n; ++i) {
        double temp = std::abs(x(i) - xold(i)) / std::max(std::abs(x(i)), 1.0);
        if (temp > test) test = temp;
      }
      if (test < kTolX) break;  // return check;

    }
    // const int kIterMax = 200; // Maximum number of iterations
    // const double kEpsilon = std::numeric_limits<double>::epsilon(); // Machine precision
    // const double kTolX = 4 * kEpsilon; // Convergence criterion on x values
    // const double kScaStepMax = 100; // Scaled maximum step length allowed in line searches
    // const double grad_tol = 1e-6;
    // const int n_pars = x.n_rows;  // number of parameters
    
    // dragonwell::NRfmin<gee::gee_jmcm> geef(gees);
    // double f = geef(x);
    // arma::vec grad;
    // grad = gees(x);
    // if (debug) grad.t().print("initial Gradient: ");
    
    // // Initialize the inverse Hessian to a unit matrix
    // arma::mat hess_inv = arma::eye<arma::mat>(n_pars, n_pars);
    
    // // Initialize Newton Step
    // arma::vec p = -hess_inv * grad;
    // if (debug) p.t().print("initial Newton Step: ");
    
    // // Initialize the maximum step length
    // double sum = sqrt(arma::dot(x, x));
    // const double kStepMax = kScaStepMax * std::max(sum, static_cast<double>(n_pars));
    
    // if (debug) Rcpp::Rcout << "Before for loop" << std::endl; 
  
    // // Main loop over the iterations
    // for (int iter = 0; iter != kIterMax; ++iter) {
    //   if (debug) Rcpp::Rcout << "iter " << iter << ":" << std::endl;
    //   n_iters = iter;
    //   arma::vec x2 = x;   // Save the old point
      
    //   linesearch.GetStep(geef, &f, &x, grad, p, kStepMax);
    //   if (debug) x.t().print("x = ");
    //   if (debug) {arma::vec tmp = (x+p); tmp.t().print("x+p = ");}

    //   if (debug) Rcpp::Rcout << "geef(x) = " << geef(x) << std::endl;
    //   if (debug) Rcpp::Rcout << "geef(x) = " << 0.5 * gees(x).t() * gees(x)  << std::endl;
    //   if (debug) Rcpp::Rcout << "geef(x + p) = " << geef(x + 0.5 * p) << std::endl;
    //   if (debug) Rcpp::Rcout << "geef(x + p) = " << 0.5 * gees(x + 0.5 * p).t() * gees(x + 0.5 * p) << std::endl;
      
    //   f = geef(x);  // Update function value
    //   p = x - x2;  // Update line direction
    //   x2 = x;
    //   //f_min = f;
      
    //   if (trace) {
    //     Rcpp::Rcout << std::setw(5) << iter << ": "
    //                 << std::setw(10) << geef(x) << ": ";
    //     x.t().print();
    //   }
      
    //   if (debug) Rcpp::Rcout << "Checking convergence..." << std::endl;
    //   // Test for convergence on Delta x
    //   double test = 0.0;
    //   for (int i = 0; i != n_pars; ++i) {
    //     double temp = std::abs(p(i)) / std::max(std::abs(x(i)), 1.0);
    //     if (temp > test) test = temp;
    //   }
      
    //   if (test < kTolX) {
    //     if (debug) {
    //       Rcpp::Rcout << "Test for convergence on Delta x: converged."
    //                   << std::endl;
    //     }
    //     break;
    //   }
      
    //   arma::vec grad2 = grad;  // Save the old gradient
    //   grad = gees(x);          // Get the new gradient
      
    //   // Test for convergence on zero gradient
    //   test = 0.0;
    //   double den = std::max(f, 1.0);
    //   for (int i = 0; i != n_pars; ++i) {
    // 	double temp = std::abs(grad(i)) * std::max(std::abs(x(i)), 1.0) / den;
    //     if (temp > test) test = temp;
    //   }
    //   if (test < grad_tol) {
    //     if (debug)
    //       Rcpp::Rcout << "Test for convergence on zero gradient: converged."
    //                   << std::endl;
    // 	break;
    //   }
      
    //   if (debug) Rcpp::Rcout << "Update beta..." << std::endl;
    //   gees.UpdateBeta();
      
    //   if (debug) Rcpp::Rcout << "Update lambda..." << std::endl;
    //   gees.UpdateLambda();
      
    //   if (debug) Rcpp::Rcout << "Update gamma..." << std::endl;
    //   gees.UpdateGamma();

    //   if (debug) Rcpp::Rcout << "Update theta..." << std::endl;
    //   arma::vec xnew = gees.get_theta();
      
    //   p = xnew - x;
    // } // for loop
  } else {
    if (debug) Rcpp::Rcout << "gees_estimation(): starting non-profile estimation..." << std::endl;
    newt.Optimize(x, 1.0e-6, trace);
    //f_min = newt.f_min();
    n_iters = newt.n_iters();
  }

  arma::vec beta   = x.rows(0, n_bta-1);
  arma::vec lambda = x.rows(n_bta, n_bta+n_lmd-1);
  arma::vec gamma  = x.rows(n_bta+n_lmd, n_bta+n_lmd+n_gma-1);

  int n_par = n_bta + n_lmd + n_gma;
  int n_sub = m.n_rows;

  double quasilik = gees.get_quasi_likelihood(x);

  return Rcpp::List::create(Rcpp::Named("par") = x,
                            Rcpp::Named("beta") = beta,
                            Rcpp::Named("lambda") = lambda,
                            Rcpp::Named("gamma") = gamma,
                            Rcpp::Named("quasilik") = quasilik,
                            Rcpp::Named("QIC") = -2 * quasilik / n_sub + 2 * n_par / n_sub,
                            Rcpp::Named("iter") = n_iters);

}


// [[Rcpp::export]]
Rcpp::List geerfit_id(arma::uvec m,
                      arma::vec Y,
                      arma::mat X,
                      arma::mat Z,
                      arma::mat W,
                      double rho,
                      arma::vec start,
                      bool trace = false,
                      bool profile = true,
                      bool errorMsg = false) {
  //int debug = 1;

  int n_bta = X.n_cols;
  int n_lmd = Z.n_cols;
  int n_gma = W.n_cols;

  gee::gee_jmcm gees(m, Y, X, Z, W, rho, identity_link, Identity_corr);
  dragonwell::Newton<gee::gee_jmcm> newt(gees);
  arma::vec x = start;

  int n_iters  = 0;

  // if (profile) {
  //   if(debug) {
  //     Rcpp::Rcout << "Start profile opt ... " << std::endl;
  //     x.print("start value: ");
  //   }
  //
  //   // Maximum number of iterations
  //   const int kIterMax = 200;
  //
  //   // Machine precision
  //   const double kEpsilon = std::numeric_limits<double>::epsilon();
  //
  //   // convergence criterion on x values
  //   const double kTolX = 4 * kEpsilon;
  //
  //   // Scaled maximum step length allowed in line searches
  //   const double kScaStepMax = 100;
  //
  //   const double grad_tol = 1e-6;
  //
  //   const int n_pars = x.n_rows;  // number of parameters
  //
  //   //double f = gees(x);
  //   arma::vec grad;
  //   grad = gees(x);
  //
  //   // Initialize the inverse Hessian to a unit matrix
  //   arma::mat hess_inv = arma::eye<arma::mat>(n_pars, n_pars);
  //
  //   // Initialize the maximum step length
  //   double sum = sqrt(arma::dot(x, x));
  //   const double kStepMax = kScaStepMax * std::max(sum, double(n_pars));
  //
  //   if (debug) Rcpp::Rcout << "Before for loop" << std::endl;
  //
  //   // Main loop over the iterations
  //   for (int iter = 0; iter != kIterMax; ++iter) {
  //     if (debug) Rcpp::Rcout << "iter " << iter << ":" << std::endl;
  //
  //     n_iters = iter;
  //
  //     arma::vec x2 = x;   // Save the old point
  //
  //
  //     p = x - x2;
  //     x2 = x;
  //
  //     if (trace) {
  //       Rcpp::Rcout << std::setw(5) << iter << ": "
  //                   << std::setw(10) << gees(x) << ": ";
  //       x.t().print();
  //
  //     }
  //
  //     if (debug) Rcpp::Rcout << "Checking convergence..." << std::endl;
  //     // Test for convergence on Delta x
  //     double test = 0.0;
  //     for (int i = 0; i != n_pars; ++i) {
  //       double temp = std::abs(p(i)) / std::max(std::abs(x(i)), 1.0);
  //       if (temp > test) test = temp;
  //     }
  //
  //     if (test < kTolX) {
  //       if (debug) {
  //         Rcpp::Rcout << "Test for convergence on Delta x: converged."
  //                     << std::endl;
  //       }
  //       break;
  //     }
  //
  //     if (debug) Rcpp::Rcout << "Update beta..." << std::endl;
  //     gees.UpdateBeta();
  //
  //     if (debug) Rcpp::Rcout << "Update lambda..." << std::endl;
  //     gees.UpdateLambda();
  //
  //     if (debug) Rcpp::Rcout << "Update gamma..." << std::endl;
  //     gees.UpdateGamma();
  //
  //   } // for loop
  //
  // }




    //double f_min = 0.0;
    newt.Optimize(x, 1.0e-6, trace);
    //f_min = newt.f_min();
    n_iters = newt.n_iters();
  //else {
  //   gees.learn(m, Y, X, Z, W, identity_link, Identity_corr, rho, x, 1000, trace);
  // }

  arma::vec beta   = x.rows(0, n_bta-1);
  arma::vec lambda = x.rows(n_bta, n_bta+n_lmd-1);
  arma::vec gamma  = x.rows(n_bta+n_lmd, n_bta+n_lmd+n_gma-1);

  int n_par = n_bta + n_lmd + n_gma;
  int n_sub = m.n_rows;

  double quasilik = gees.get_quasi_likelihood(x);

  return Rcpp::List::create(Rcpp::Named("par") = x,
                            Rcpp::Named("beta") = beta,
                            Rcpp::Named("lambda") = lambda,
                            Rcpp::Named("gamma") = gamma,
                            Rcpp::Named("quasilik") = quasilik,
                            Rcpp::Named("QIC") = -2 * quasilik / n_sub + 2 * n_par / n_sub,
                            Rcpp::Named("iter") = n_iters);

}

// [[Rcpp::export]]
Rcpp::List geerfit_cs(arma::uvec m,
                          arma::vec Y,
                          arma::mat X,
                          arma::mat Z,
                          arma::mat W,
                          double rho,
                          arma::vec start,
                          bool trace = false,
                          bool profile = true,
                          bool errorMsg = false) {
  int n_bta = X.n_cols;
  int n_lmd = Z.n_cols;
  int n_gma = W.n_cols;

  gee::gee_jmcm gees(m, Y, X, Z, W, rho, identity_link, CompSymm_corr);
  dragonwell::Newton<gee::gee_jmcm> newt(gees);
  arma::vec x = start;

  //double f_min = 0.0;
  int n_iters  = 0;

  newt.Optimize(x, 1.0e-6, trace);
  //f_min = newt.f_min();
  n_iters = newt.n_iters();

  arma::vec beta   = x.rows(0, n_bta-1);
  arma::vec lambda = x.rows(n_bta, n_bta+n_lmd-1);
  arma::vec gamma  = x.rows(n_bta+n_lmd, n_bta+n_lmd+n_gma-1);

  int n_par = n_bta + n_lmd + n_gma;
  int n_sub = m.n_rows;

  double quasilik = gees.get_quasi_likelihood(x);

  return Rcpp::List::create(Rcpp::Named("par") = x,
                            Rcpp::Named("beta") = beta,
                            Rcpp::Named("lambda") = lambda,
                            Rcpp::Named("gamma") = gamma,
                            Rcpp::Named("quasilik") = quasilik,
                            Rcpp::Named("QIC") = -2 * quasilik / n_sub + 2 * n_par / n_sub,
                            Rcpp::Named("iter") = n_iters);

}

// [[Rcpp::export]]
Rcpp::List geerfit_ar1(arma::uvec m,
                   arma::vec Y,
                   arma::mat X,
                   arma::mat Z,
                   arma::mat W,
                   double rho,
                   arma::vec start,
                   bool trace = false,
                   bool profile = true,
                   bool errorMsg = false) {
  int n_bta = X.n_cols;
  int n_lmd = Z.n_cols;
  int n_gma = W.n_cols;

  gee::gee_jmcm gees(m, Y, X, Z, W, rho, identity_link, AR1_corr);
  dragonwell::Newton<gee::gee_jmcm> newt(gees);
  arma::vec x = start;

  //double f_min = 0.0;
  int n_iters  = 0;

  newt.Optimize(x, 1.0e-6, trace);
  //f_min = newt.f_min();
  n_iters = newt.n_iters();

  arma::vec beta   = x.rows(0, n_bta-1);
  arma::vec lambda = x.rows(n_bta, n_bta+n_lmd-1);
  arma::vec gamma  = x.rows(n_bta+n_lmd, n_bta+n_lmd+n_gma-1);

  int n_par = n_bta + n_lmd + n_gma;
  int n_sub = m.n_rows;

  double quasilik = gees.get_quasi_likelihood(x);

  return Rcpp::List::create(Rcpp::Named("par") = x,
                            Rcpp::Named("beta") = beta,
                            Rcpp::Named("lambda") = lambda,
                            Rcpp::Named("gamma") = gamma,
                            Rcpp::Named("quasilik") = quasilik,
                            Rcpp::Named("QIC") = -2 * quasilik / n_sub + 2 * n_par / n_sub,
                            Rcpp::Named("iter") = n_iters);

}
