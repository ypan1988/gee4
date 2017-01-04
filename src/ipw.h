#ifndef IPW_H_
#define IPW_H_

#include <RcppArmadillo.h>

namespace gee {

  class ipw {
  public:
  ipw(const arma::uvec &m, const arma::vec &Y, arma::uword order)
    : m_(m), Y_(Y), order_(order) { }

    inline arma::vec get_Y(const arma::uword i) const {
      arma::vec Yi;
      if (i == 0)
        Yi = Y_.subvec(0, m_(0) - 1);
      else {
        arma::uword vindex = arma::sum(m_.subvec(0, i - 1));
        Yi = Y_.subvec(vindex, vindex + m_(i) - 1);
      }
      return Yi;
    }
        
    arma::vec operator()(const arma::vec &alpha) {
      arma::uword nsub = m_.n_elem;
      arma::uword npar = alpha.n_elem;

      arma::vec result = arma::zeros<arma::vec>(npar);
      for (auto i = 0; i != nsub; ++i) {
        arma::vec Yi = get_Y(i);
        for (auto j = 1; j != m_(i); ++j) {
          arma::vec Z_ij = arma::zeros<arma::vec>(npar);
          if ((j+1) <= order_) Z_ij(0) = 1;
          else {
            Z_ij(0) = 1;
            for (auto k = 1; k <= order_; ++k) Z_ij(k) = Yi(j - k); 
          } 
          
          double tmp = arma::as_scalar(arma::exp(Z_ij.t() * alpha));
          double p_ij =  tmp / (1 + tmp);
          result += 1 * (1 - p_ij) * Z_ij;
        }
      }

      return result;
    }

    arma::vec CalWeights(const arma::vec &alpha) {
      int debug = 1;
      
      arma::uword nsub = m_.n_elem;
      arma::uword npar = alpha.n_elem;
      
      arma::uword index = 0;
      arma::vec result = arma::zeros<arma::vec>(Y_.n_elem);
      for (auto i = 0; i != nsub; ++i) {
	arma::vec Yi = get_Y(i);
	arma::vec p_i = arma::zeros<arma::vec>(m_(i));
	arma::vec Pi_i = arma::zeros<arma::vec>(m_(i));
	for (auto j = 0; j != m_(i); ++j, ++index) {
	  
	  if (j == 0) {
	    p_i(0) = 0;
	    Pi_i(0) = 1 - p_i(0);
	    result(index) = 1 / Pi_i(0);
	  } else {
	    arma::vec Z_ij = arma::zeros<arma::vec>(npar);
	    if ((j+1) <= order_) Z_ij(0) = 1;
	    else {
	      Z_ij(0) = 1;
	      for (auto k = 1; k <= order_; ++k) Z_ij(k) = Yi(j - k); 
	    } 
	    
	    double tmp = arma::as_scalar(arma::exp(Z_ij.t() * alpha));
	    double p_ij =  tmp / (1 + tmp);
	    p_i(j) = p_ij;
	    Pi_i(j) = Pi_i(j-1) * (1 - p_ij);
	    result(index) = result(index-1) / Pi_i(j);
	  }

	  if (debug) Rcpp::Rcout << index << ": " << result(index) << " ";  
	} // for loop j
	if (debug) Rcpp::Rcout << std::endl;
      } // for loop i
      return result;
    } 
    
  private:
    arma::uvec m_;
    arma::vec Y_;
    arma::uword order_;
  };
}

#endif
