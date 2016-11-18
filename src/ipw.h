#ifndef IPW_H_
#define IPW_H_

#include <RcppArmadillo.h>

namespace gee {

  class ipw {
  public:
    ipw(const arma::uvec &m, const arma::vec &Y, const arma::X) {
    }

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
    
    inline arma::vec get_R(const arma::uword i) const {
      arma::vec Ri;
      if (i == 0) Ri = R_.subvec(0, m_(0)-1);
      else {
	arma::uword vindex = arma::sum(m_.subvec(0, i - 1));
	Ri = R_.subvec(vindex, vindex + m_(i) - 1);
      }
      return Ri;
    } 
    
    arma::vec operator()(const arma::vec &alpha) {
      arma::uword nsub = m.n_elem;

      arma::vec result = arma::zeros<arma::vec>(x.n_elem);
      for (auto i = 0; i != n_sub; ++i) {
	arma::vec Yi = get_Y(i);
	arma::vec Ri = get_R(i);
	for (auto j = 1; j != m(i); ++j) {
	  arma::vec Z_ij = {1, Yi(j-1)};
	  double tmp = arma::as_scalar(arma::exp(Z_ij * alpha));
	  double p_ij =  tmp / (1 + tmp);
	  result += Ri(j-1) * (Ri(j) - p_ij) * Z_ij;
	}
      }

      return result;
    }
    
  private:
    arma::uvec m_;
    arma::vec Y_;
    arma::vec R_;
    arma::mat X_;
    
    arma::vec alpha_;
  }
}

#endif
