#include "LogDetTriDiag.hpp"



/*
 Compute Log-Determinant of N Tri-diagonal matrices simultaneously
 Yu Hang, NTU, Jul 2019
 Inputs:
           K_d:  P x N matrix of the diagonals of the N P x P tri-diagonal matrices
           K_od: P-1 x N matrix of the NEGATIVE first off-diagonals of the N P x P tri-diagonal matrices
*/

logdet_result LogDetTriDiag(mat K_d, mat K_od, uword P, uword N) {

  rowvec logdetK_vec(N, fill::zeros), C_d2, C_od;
  bool is_pd = true;
  
  
  for (uword j = 0; j < P; j++) {
    if (j == 0) {
      C_d2 = K_d.row(j);
    }
    else {
      C_od = K_od.row(j-1) / sqrt(C_d2);
      C_d2 = K_d.row(j) - square(C_od);
    }
    if (any(C_d2 <= 0)) {
      is_pd = false;
      break;
    } else {
      logdetK_vec += log(C_d2);
    }
  }
  return {logdetK_vec, is_pd};
}