#include "CrossValidation.hpp"

double CrossValidation(cx_mat data, uword n_fold, uword N, uword P) {
    
    double cv_score_old = -1e100, cv_score, bandwidth = 2.0, det_S, bd_max;
    uword iter, i, j, id_r, id_c;
    vec id_all = regspace(0, N - 1), weights, weights0, id_w0, idj, S_diag;
    uvec id_validation, id_w, id_sub, id_vec(N);
    cx_mat S_all;
    bool is_cv_old = false, is_cv;
    
    for (iter = 0; iter < N - 2; iter ++) {
        is_cv = true;
        cv_score = 0;
        id_w0 = regspace(-(ceil(bandwidth) - 1), ceil(bandwidth) - 1);
        weights0 = 1 - square(id_w0 / bandwidth);
        for (i = 0; i < n_fold; i ++) {
            id_vec.fill(1);
            id_validation = regspace<uvec>(i, n_fold, N - 1);
            id_vec(id_validation).zeros();
            for (j = 0; j < id_validation.n_elem; j ++) {
                idj = id_w0 + id_validation(j);
                id_sub = find((idj >= 0) % (idj < N));
                weights = weights0(id_sub);
                id_w = conv_to<uvec>::from(idj(id_sub));
                id_sub.clear();
                id_sub = find(id_vec(id_w) > 0);
                
                if (id_sub.n_elem < 2) {
                    is_cv = false;
                    
                    id_sub.clear();
                    id_w.clear();
                    weights.clear();
                    
                    break;
                } else{
                    // S = E[xx^H], where x is a column vector
                    S_all = data.rows(id_w(id_sub)).st() % repmat(weights(id_sub).t() / sum(weights(id_sub)), P, 1) *
                            conj(data.rows(id_w(id_sub)));
                    S_diag = real(S_all.diag());
                    id_sub.clear();
                    id_w.clear();
                    weights.clear();
                    
                    // log_pdf = - x^H K x / 2
                    for (id_r = 0; id_r < P - 1; id_r ++) {
                        for (id_c = id_r + 1; id_c < P; id_c ++) {
                            det_S = S_diag(id_r) * S_diag(id_c) - real(S_all(id_r, id_c) * S_all(id_c, id_r));
                            cv_score -= log(det_S) + real(data(id_validation(j), id_r) * conj(data(id_validation(j), id_r)) * S_diag(id_c) +
                                    data(id_validation(j), id_c) * conj(data(id_validation(j), id_c)) * S_diag(id_r) -
                                    2.0 * conj(data(id_validation(j), id_r)) * data(id_validation(j), id_c) * S_all(id_r, id_c)) / det_S;
                        }
                    }
                }
                
            }
            id_validation.clear();
        }
        if (is_cv_old && is_cv && cv_score < cv_score_old) {
            // printf("cv_score = %f, cv_score_old = %f, \n", cv_score, cv_score_old);
            break;
        } else {
            if (is_cv) {
                if (cv_score > cv_score_old) bd_max = bandwidth;
                cv_score_old = cv_score;
                is_cv_old = true;
            }
            bandwidth *= 1.5;
            weights0.clear();
            id_w0.clear();
            idj.clear();
        }
    }
    
    

    return bd_max;
    
    
}