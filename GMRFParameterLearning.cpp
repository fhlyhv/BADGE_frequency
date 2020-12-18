#include "GMRFParameterLearning.hpp"

cx_mat GMRFParameterLearning(cx_mat K0, cx_mat S, uvec idr, uvec idc, uword max_iter) {
  uword p = K0.n_cols, iter, p_od = idr.n_elem, i;
  uvec idl = idc * p + idr, idu = idr * p + idc, idd = regspace<uvec>(0, p - 1), ida = join_cols(idd * p + idd, idl);
  cx_vec Sida = S(ida), Sidu = S(idu), gradK, Kd = K0.diag();
  vec Sd = real(S.diag());
  cx_mat Kh(p, p, fill::zeros), W0(p, p, fill::zeros), Wh(p, p, fill::zeros);
  S.clear();
  
  double objh, xi0 = 1, xi, sum_grad2;
  while (! K0.is_sympd()) {
    K0 *= 0.9;
    K0.diag() = Kd;
  }
  double obj0 = - 2*accu(log(real(diagvec(chol(K0))))) + accu(Sd % real(K0.diag()))+ 2 * accu(real(Sidu % K0(idl)));
  W0 = inv_sympd(K0);
//   mexPrintf("obj0 = %f\n", obj0);
//   mexEvalString("drawnow;");
  for (iter = 0; iter < max_iter; iter ++) {
    gradK = Sida - W0(ida);
    sum_grad2 = accu(real(gradK % conj(gradK)));
    xi = xi0;
    while (xi > 1e-20) {
      Kh.diag() = K0.diag() - xi * real(gradK.head(p));
      Kh(idl) = K0(idl) - xi * gradK.tail(p_od);
      Kh(idu) = conj(Kh(idl));
//       mexPrintf("xi = %e\n", xi);
//       mexEvalString("drawnow;");
      if (Kh.is_sympd()) {
        objh = - 2*accu(log(real(diagvec(chol(Kh))))) + accu(Sd % real(Kh.diag())) + 2 * accu(real(Sidu % Kh(idl)));
//         mexPrintf("objh = %f\n", objh);
//         mexEvalString("drawnow;");
        if (objh <= obj0 + 1e-3 * xi * sum_grad2)
          break;
        else
          xi /= 2;
      } else
        xi /=2;
    }
    if (xi <= 1e-20) {
        xi = 0;
        Kh(ida) = K0(ida);
        Kh(idu) = K0(idu);
        obj0 = objh;
    }
    // printf("xi = %e\n", xi);
    if (xi * abs(gradK).max() < 1e-6 && obj0 - objh < 1e-6) {
      break;
    } else {
      Wh = inv_sympd(Kh);
//       mexPrintf("iter = %i, xi0 = %e, xi = %e, objh = %f, obj0 - objh = %e\n", iter, xi0, xi, objh, obj0 - objh);
//       mexEvalString("drawnow;");
      xi0 = abs(accu(real(gradK % conj(W0(ida) - Wh(ida)))));
      if (xi0 > 0) xi0 = xi * sum_grad2 / xi0 ;
      else xi0 = xi;
      //if (xi0 > 1) xi0 = 1;
      obj0 = objh;
      K0(ida) = Kh(ida);
      K0(idu) = Kh(idu);
      W0(ida) = Wh(ida);
    }
  }
//   mexPrintf("The algorithm converges, iter = %i\n", iter);
//   mexEvalString("drawnow;");
  
  return Kh;
}