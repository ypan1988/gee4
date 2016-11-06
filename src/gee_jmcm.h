#ifndef GEE_JMCM_H_
#define GEE_JMCM_H_

#include "utils.h"
#include <RcppArmadillo.h>

struct gee_link_mode {
  const arma::uword id_;
  inline explicit gee_link_mode(const arma::uword id) : id_(id) {}
};

inline bool operator==(const gee_link_mode &a, const gee_link_mode &b) {
  return (a.id_ == b.id_);
}

inline bool operator!=(const gee_link_mode &a, const gee_link_mode &b) {
  return (a.id_ != b.id_);
}

struct gee_link_identity : public gee_link_mode {
  inline gee_link_identity() : gee_link_mode(1) {}
};

static const gee_link_identity identity_link;

struct gee_corr_mode {
  const arma::uword id_;
  inline explicit gee_corr_mode(const arma::uword id) : id_(id) {}
};

inline bool operator==(const gee_corr_mode &a, const gee_corr_mode &b) {
  return (a.id_ == b.id_);
}

inline bool operator!=(const gee_corr_mode &a, const gee_corr_mode &b) {
  return (a.id_ != b.id_);
}

struct gee_corr_Identity : public gee_corr_mode {
  inline gee_corr_Identity() : gee_corr_mode(1) {}
};

struct gee_corr_CompSymm : public gee_corr_mode {
  inline gee_corr_CompSymm() : gee_corr_mode(2) {}
};

struct gee_corr_AR1 : public gee_corr_mode {
  inline gee_corr_AR1() : gee_corr_mode(3) {}
};

static const gee_corr_Identity Identity_corr;
static const gee_corr_CompSymm CompSymm_corr;
static const gee_corr_AR1 AR1_corr;

namespace gee {

class gee_jmcm {
 public:
  //gee_jmcm(){link_mode_ = identity_link};
   gee_jmcm(const arma::uvec &m,
            const arma::vec &Y,
            const arma::mat &X,
            const arma::mat &Z,
            const arma::mat &W,
            const double rho,
            const gee_link_mode& link_mode,
            const gee_corr_mode &corr_mode):
     m_(m),Y_(Y),X_(X),Z_(Z),W_(W), rho_(rho),
     link_mode_(link_mode), corr_mode_(corr_mode){

   }
  ~gee_jmcm(){};

  inline void set_params(const arma::vec &x);

  arma::vec operator()(const arma::vec &x);

  void UpdateGEES(const arma::vec &x);
  void UpdateParam(const arma::vec &x);
  void UpdateModel();
  
  void UpdateBeta();
  void UpdateLambda();
  void UpdateGamma();

  bool learn(const arma::uvec &m, const arma::mat &Y, const arma::mat &X,
             const arma::mat &Z, const arma::mat &W,
             const gee_link_mode &link_mode, const gee_corr_mode &corr_mode,
             const double rho, const arma::vec &start,
             const arma::uword fs_iter, const bool print_mode = false);

  inline double get_quasi_likelihood(const arma::vec &x) {
    set_params(x);
    arma::uword n_sub = m_.n_elem;

    double result = 0.0;
    for(arma::uword i = 0; i < n_sub; ++i) {
      arma::mat Sigmai_inv = get_Sigma_inv(i);
      arma::vec ri = get_Resid(i);
      result += arma::as_scalar(ri.t() * Sigmai_inv * ri);
      //result += arma::as_scalar(ri.t() * ri);
    }
    result *= -0.5;

    return result;
  }

 private:
  arma::uvec m_;
  arma::vec Y_;
  arma::mat X_;
  arma::mat Z_;
  arma::mat W_;

  arma::vec tht_;
  arma::vec bta_;
  arma::vec lmd_;
  arma::vec gma_;

  arma::vec Xbta_;
  arma::vec Zlmd_;
  arma::vec Wgma_;
  arma::vec Resid_;

  int free_param_;

  double rho_;
  gee_link_mode link_mode_;
  gee_corr_mode corr_mode_;

  inline arma::uword get_m(const arma::uword i) const;
  inline arma::vec get_Y(const arma::uword i) const;
  inline arma::mat get_X(const arma::uword i) const;
  inline arma::mat get_Z(const arma::uword i) const;
  inline arma::mat get_W(const arma::uword i) const;

  inline arma::vec get_Resid(const arma::uword i) const;
  inline arma::mat get_D(const arma::uword i) const;
  inline arma::mat get_T(const arma::uword i) const;
  inline arma::mat get_Sigma_inv(const arma::uword i) const;

  bool fs_iterate(const gee_link_mode &link_mode,
                  const gee_corr_mode &corr_mode, const double rho,
                  const arma::vec &start, const arma::uword max_iter,
                  const bool verbose);
  void fs_update_params(const gee_link_mode &link_mode,
                        const gee_corr_mode &corr_mode, const double rho);
};

void gee_jmcm::set_params(const arma::vec &x) {
  int debug = 0;
  if (debug) Rcpp::Rcout << "inside of set_params" << std::endl;
  arma::uword lbta = X_.n_cols;
  arma::uword llmd = Z_.n_cols;
  arma::uword lgma = W_.n_cols;
  if (x.n_elem != (lbta + llmd + lgma))
    Rcpp::Rcerr << "gee_jmcm::set_params(): give vector has wrong dimension"
              << std::endl;

  bta_ = x.rows(0, lbta - 1);
  lmd_ = x.rows(lbta, lbta + llmd - 1);
  gma_ = x.rows(lbta + llmd, lbta + llmd + lgma - 1);

  if (debug) Rcpp::Rcout << "set_param update" << std::endl;

  Xbta_ = X_ * bta_;
  Zlmd_ = Z_ * lmd_;
  Wgma_ = W_ * gma_;
  Resid_ = Y_ - Xbta_;
}

arma::uword gee_jmcm::get_m(const arma::uword i) const { return m_(i); }

arma::vec gee_jmcm::get_Y(const arma::uword i) const {
  arma::vec Yi;
  if (i == 0)
    Yi = Y_.subvec(0, m_(0) - 1);
  else {
    arma::uword vindex = arma::sum(m_.subvec(0, i - 1));
    Yi = Y_.subvec(vindex, vindex + m_(i) - 1);
  }
  return Yi;
}

arma::mat gee_jmcm::get_X(const arma::uword i) const {
  arma::mat Xi;
  if (i == 0)
    Xi = X_.rows(0, m_(0) - 1);
  else {
    arma::uword rindex = arma::sum(m_.subvec(0, i - 1));
    Xi = X_.rows(rindex, rindex + m_(i) - 1);
  }
  return Xi;
}

arma::mat gee_jmcm::get_Z(const arma::uword i) const {
  arma::mat Zi;
  if (i == 0)
    Zi = Z_.rows(0, m_(0) - 1);
  else {
    arma::uword rindex = arma::sum(m_.subvec(0, i - 1));
    Zi = Z_.rows(rindex, rindex + m_(i) - 1);
  }
  return Zi;
}

arma::mat gee_jmcm::get_W(const arma::uword i) const {
  arma::mat Wi;
  if (m_(i) != 1) {
    if (i == 0) {
      arma::uword first_index = 0;
      arma::uword last_index = m_(0) * (m_(0) - 1) / 2 - 1;
      Wi = W_.rows(first_index, last_index);
    } else {
      arma::uword first_index = 0;
      for (arma::uword idx = 0; idx != i; ++idx) {
        first_index += m_(idx) * (m_(idx) - 1) / 2;
      }
      arma::uword last_index = first_index + m_(i) * (m_(i) - 1) / 2 - 1;
      Wi = W_.rows(first_index, last_index);
    }
  }
  return Wi;
}

arma::vec gee_jmcm::get_Resid(const arma::uword i) const {
  arma::vec ri;
  if (i == 0)
    ri = Resid_.subvec(0, m_(0) - 1);
  else {
    arma::uword vindex = arma::sum(m_.subvec(0, i - 1));
    ri = Resid_.subvec(vindex, vindex + m_(i) - 1);
  }
  return ri;
}

arma::mat gee_jmcm::get_D(const arma::uword i) const {
  arma::mat Di = arma::eye(m_(i), m_(i));
  if (i == 0)
    Di = arma::diagmat(arma::exp(Zlmd_.subvec(0, m_(0) - 1)));
  else {
    arma::uword rindex = arma::sum(m_.subvec(0, i - 1));
    Di = arma::diagmat(arma::exp(Zlmd_.subvec(rindex, rindex + m_(i) - 1)));
  }
  return Di;
}

arma::mat gee_jmcm::get_T(const arma::uword i) const {
  arma::mat Ti = arma::eye(m_(i), m_(i));
  if (m_(i) != 1) {
    if (i == 0) {
      arma::uword first_index = 0;
      arma::uword last_index = m_(0) * (m_(0) - 1) / 2 - 1;
      Ti = dragonwell::ltrimat(m_(0), -Wgma_.subvec(first_index, last_index));
    } else {
      arma::uword first_index = 0;
      for (arma::uword idx = 0; idx != i; ++idx) {
        first_index += m_(idx) * (m_(idx) - 1) / 2;
      }
      arma::uword last_index = first_index + m_(i) * (m_(i) - 1) / 2 - 1;
      Ti = dragonwell::ltrimat(m_(i), -Wgma_.subvec(first_index, last_index));
    }
  }
  return Ti;
}

arma::mat gee_jmcm::get_Sigma_inv(const arma::uword i) const {
  arma::mat Ti = get_T(i);
  arma::mat Di = get_D(i);
  arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));
  return Ti.t() * Di_inv * Ti;
}

}  // namespace gee4

#endif  // GEE_JMCM_H_
