#include "SumEKodMDataUpdate.hpp"

cx_mat SumEKodMDataUpdate(cx_mat data, cx_mat EKod_mat, uvec idl, uvec idu,
        umat idl_mat, uword N, uword P) {
    
    uword i, j;
    uvec vP = regspace<uvec>(0, P - 1), id_tmp;
    cx_mat EKodi, sum_EKod_m_data(N, P);
    
    if (P < N) {
        for (j = 0; j < P; j ++) {
           id_tmp = join_cols(vP.head(j), vP.tail(P - 1 - j));
            if (j == 0) EKodi = conj(EKod_mat.cols(idl_mat.row(j)));
            else if (j == P - 1) EKodi = EKod_mat.cols(idl_mat.row(j));
            else EKodi = join_rows(EKod_mat.cols(idl_mat(j, span(0, j))),
                    conj(EKod_mat.cols(idl_mat(j, span(j + 1, P - 1)))));
            sum_EKod_m_data.col(j) = sum(data.cols(id_tmp) % EKodi.cols(id_tmp), 1);
        }
    } else {
        EKodi.set_size(P, P);
        EKodi.zeros();
        for (i = 0; i < N; i ++) {
            EKodi(idl) = conj(EKod_mat.row(i));
            EKodi(idu) = EKod_mat.row(i);   // EKodi.st()
            sum_EKod_m_data.row(i) = data.row(i) * EKodi;
        }
    }
    
    return sum_EKod_m_data;
}