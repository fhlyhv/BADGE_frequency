#include "GMRFParameterLearning_diag.hpp"

cx_mat GMRFParameterLearning_diag(cx_mat K0, vec S_diag, uword max_iter) {
  uword p = K0.n_cols, iter, i;
  uvec idl = idc * p + idr, idu = idr * p + idc, idd = regpace<uvec>(0, p - 1), ida = join_cols(idd * p + idd, idl);
  cx_vec gradK;
  cx_mat Kh(p, p, fill::zeros), W0(p, p, fill::zeros), Wh(p, p, fill::zeros);
  
  
  double objh, xi0 = 1, xi, sum_grad2;
  while (! K0.is_sympd()) {
    K0 *= 0.9;
    K0.diag() = Kd;
  }
  double obj0 = - 2*accu(log(real(diagvec(chol(K0))))) + accu(real(Sida % K0(ida)))+ accu(real(Sidu % K0(idu)));
  W0(ida) = inv_sympd(K0).st().elem(ida);
  
  for (iter = 0; iter < max_iter; iter ++) {
    gradK = Sida - W0(ida);
    sum_grad2 = accu(real(gradK % conj(gradK)));
    xi = xi0;
    while (xi > 1e-10) {
      Kh.diag() = K0.diag() + xi * real(gradK.head(p));
      Kh(idl) = K0(idl) + xi * gradK.tail(p_od);
      Kh(idu) = conj(Kh(idl));
      if (Kh.is_sympd()) {
        objh = - 2*accu(log(real(diagvec(chol(Kh))))) + accu(real(Sida % Kh(ida))) + accu(real(Sidu % Kh(idu)));
        if (objh <= obj0 + 1e-3 * xi * sum_grad2)
          break;
        else
          xi /= 2;
      } else
        xi /=2;
    }
    if (xi <= 1e-12) {
        xi = 0;
        Kh(ida) = K0(ida);
        Kh(idu) = K0(idu);
        obj0 = objh;
    }
    // printf("xi = %e\n", xi);
    if (abs(Kh - K0).max() < 1e-10 && fabs(obj0 - objh ) < 1e-10) {
      break;
    } else {
      Wh(ida) = inv_sympd(Kh).st().elem(ida);
      xi0 = xi * accu(real(gradK % conj(gradK))) / accu(square(abs(gradK % (W0(ida) - Wh(ida))))) ;
      obj0 = objh;
      K0(ida) = Kh(ida);
      K0(idu) = Kh(idu);
      W0(ida) = Wh(ida);
    }
  }
  return Kh;
}