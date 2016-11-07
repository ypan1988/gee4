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
  
  int n_bta = X.n_cols;
  int n_lmd = Z.n_cols;
  int n_gma = W.n_cols;

  gee_corr_mode corr_mode(0);
  if (corrStruct == "id") corr_mode.setid(1);
  else if (corrStruct == "cs") corr_mode.setid(2);
  else if (corrStruct == "ar1") corr_mode.setid(3);
  
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
    Rcpp::Rcout << "Start profile optimization ... " << std::endl;
    x.print("start value: ");
  }
  
  const int kIterMax = 200; // Maximum number of iterations
  const double kEpsilon = std::numeric_limits<double>::epsilon(); // Machine precision
  const double kTolX = 4 * kEpsilon; // Convergence criterion on x values
  const double kScaStepMax = 100; // Scaled maximum step length allowed in line searches
  const double grad_tol = 1e-6;
  const int n_pars = x.n_rows;  // number of parameters
 
  dragonwell::NRfmin<gee::gee_jmcm> geef(gees);
  double f = geef(x);
  arma::vec grad;
  grad = gees(x); 

  // Initialize the inverse Hessian to a unit matrix
  arma::mat hess_inv = arma::eye<arma::mat>(n_pars, n_pars);

  // Initialize Newton Step
  arma::vec p = -hess_inv * grad;
  
  // Initialize the maximum step length
  double sum = sqrt(arma::dot(x, x));
  const double kStepMax = kScaStepMax * std::max(sum, static_cast<double>(n_pars));
  
  if (debug) Rcpp::Rcout << "Before for loop" << std::endl; 
  
  // Main loop over the iterations
  for (int iter = 0; iter != kIterMax; ++iter) {
    if (debug) Rcpp::Rcout << "iter " << iter << ":" << std::endl;
      n_iters = iter;
      arma::vec x2 = x;   // Save the old point

      linesearch.GetStep(geef, x, p, kStepMax);
      
      f = geef(x);  // Update function value
      p = x - x2;  // Update line direction
      x2 = x;
      //f_min = f;

      if (trace) {
        Rcpp::Rcout << std::setw(5) << iter << ": "
                    << std::setw(10) << geef(x) << ": ";
        x.t().print();
      }

      if (debug) Rcpp::Rcout << "Checking convergence..." << std::endl;
      // Test for convergence on Delta x
      double test = 0.0;
      for (int i = 0; i != n_pars; ++i) {
        double temp = std::abs(p(i)) / std::max(std::abs(x(i)), 1.0);
        if (temp > test) test = temp;
      }
      
      if (test < kTolX) {
        if (debug) {
          Rcpp::Rcout << "Test for convergence on Delta x: converged."
                      << std::endl;
        }
        break;
      }
 
      arma::vec grad2 = grad;  // Save the old gradient
      grad = gees(x);          // Get the new gradient
  
      // Test for convergence on zero gradient
      test = 0.0;
      double den = std::max(f, 1.0);
      for (int i = 0; i != n_pars; ++i) {
      double temp = std::abs(grad(i)) * std::max(std::abs(x(i)), 1.0) / den;
        if (temp > test) test = temp;
      }
      if (test < grad_tol) {
        if (debug)
          Rcpp::Rcout << "Test for convergence on zero gradient: converged."
                      << std::endl;
          break;
      }
      
      if (debug) Rcpp::Rcout << "Update beta..." << std::endl;
      gees.UpdateBeta();

      if (debug) Rcpp::Rcout << "Update lambda..." << std::endl;
      gees.UpdateLambda();
 
      if (debug) Rcpp::Rcout << "Update gamma..." << std::endl;
      gees.UpdateGamma();
  } // for loop

  ///////////////////////////////
  
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
