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

      arma::vec result = arma::zeros<arma::vec>(alpha.n_elem);
      for (auto i = 0; i != nsub; ++i) {
        arma::vec Yi = get_Y(i);
        for (auto j = 1; j != m_(i); ++j) {
          arma::vec Z_ij = {1, Yi(j-1)};
          double tmp = arma::as_scalar(arma::exp(Z_ij * alpha));
          double p_ij =  tmp / (1 + tmp);
          result += 1 * (1 - p_ij) * Z_ij;
        }
      }

      return result;
    }
    
  private:
    arma::uvec m_;
    arma::vec Y_;
    arma::uword order_;
  };
}

#endif
