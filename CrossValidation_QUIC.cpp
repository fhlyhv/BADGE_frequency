#include "CrossValidation_QUIC.hpp"


field<cx_mat> CrossValidation_QUIC(cx_mat data, uvec idr, uvec idc, uvec idl, uvec idu, uword N, uword P, uword Pe) {
    double bandwidth_init = (P / 2.0 < N / 10.0) ? P / 2.0 : N / 10.0;
    double cv_score_old = - datum::inf, cv_score, bandwidth = bandwidth_init, cv_max = cv_score_old, bd_max, thr;
    vec id_w0, weights0, w, idi, std_vec, cdf_s(1);
    uvec id_sub, id_w, id_nonzero, id_zero, idi_vec(1);
    cx_mat K, S_normalized, EJ_mat(N, Pe), K_init;
    mat Es_mat(N, Pe), Es_mat0, Es_old, EKd_mat(N, P), w_mat, I(P, P, fill::eye);
    cx_cube S(P, P, N), K_cube(P, P, N);
    bool is_cv_score;
    field<cx_mat> outputs(3);
    int i;
    cdf_s.fill(1.0 - (double) P / Pe); //(0.7); //
    
    
    K_cube.slice(0) = cx_mat(I, zeros(P, P));
    
    while (bandwidth < N) {
        cv_score = 0;
        is_cv_score = true;
        
        id_w0 = join_cols(regspace(- ceil(bandwidth) + 1.0, -1.0), regspace(1.0, ceil(bandwidth) - 1.0));
        weights0 = 1.0 - square(id_w0 / bandwidth);
        weights0 /= accu(weights0);
        
        
        
        for (i = 0; i < N; i ++) {
            idi = id_w0 + i;
            id_sub = find((idi >=0) % (idi < N));
            if (id_sub.n_elem < idi.n_elem) {
                w = weights0(id_sub);
                w /= accu(w);
                id_w = conv_to<uvec>::from(idi(id_sub));
            } else {
                w = weights0;
                id_w = conv_to<uvec>::from(idi);
            }
            
            S.slice(i) = (data.rows(id_w) % repmat(w, 1, P)).st() * conj(data.rows(id_w));
            /*std_vec = sqrt(real(S.slice(i).diag()));

            if (std_vec.min() <= 0) {
                is_cv_score = false;
                break;
            }
            
            S_normalized = S.slice(i) / (std_vec * std_vec.t());
            Es_mat.row(i) = abs(S_normalized(idl)).t();*/
            
            S_normalized = S.slice(i);
            S_normalized.diag() += 1.0;
            K = inv_sympd(S_normalized);
            Es_mat.row(i) = abs(K(idl)).t();
        }
        
        
        
        if (is_cv_score) {
            thr = as_scalar(quantile(vectorise(Es_mat), cdf_s));
            Es_mat.clean(thr); //(find(Es_mat < thr)).zeros();
            Es_mat(find(Es_mat >= thr)).ones();
            
            
            
            for (i = 0; i < N; i ++) {
                id_nonzero = find(Es_mat.row(i));
                if (id_nonzero.n_elem > 0) {
                    if (bandwidth == bandwidth_init) {
                        if (i > 0) {
                            K_cube.slice(i) = K_cube.slice(i - 1);
                            id_zero = find(Es_mat.row(i) - Es_mat.row(i - 1) < 0);
                            if (id_zero.n_elem > 0) {
                                K_cube.slice(i)(idl(id_zero)).zeros();
                                K_cube.slice(i)(idu(id_zero)).zeros();
                            }
                        }
                    } else {
                        id_zero = find(Es_mat.row(i) - Es_mat0.row(i) < 0);
                        if (id_zero.n_elem > 0) {
                            K_cube.slice(i)(idl(id_zero)).zeros();
                            K_cube.slice(i)(idu(id_zero)).zeros();
                        }
                    }
                    
                    //if (i == 273) K_init.print();
                    K = GMRFParameterLearning(K_cube.slice(i), S.slice(i), idr(id_nonzero), idc(id_nonzero), 1000);
                    cv_score += accu(log(real(diagvec(chol(K))))) - as_scalar(real(conj(data.row(i)) * K * data.row(i).st())) / 2;
                } else {
                    K = diagmat(1 / diagvec(S.slice(i)));
                    cv_score += (accu(log(real(K.diag()))) - as_scalar(real(data.row(i) % conj(data.row(i))) * real(K.diag()))) / 2;
                }
                
                K_cube.slice(i) = K;
                
            }
            
            mexPrintf("bandwidth = %f, cv_score = %f, cv_score_old = %f\n", bandwidth, cv_score, cv_score_old);
            mexEvalString("drawnow;");
            if (cv_score < cv_score_old && bandwidth >= P) break;  //&& bandwidth >= P
            else {
                if (cv_score > cv_max) {
                    cv_max = cv_score;
                    bd_max = bandwidth;
                }
                Es_mat0 = Es_mat;
                cv_score_old = cv_score;
                bandwidth *= 1.5;
            }
        }
    }
    
    Es_old = Es_mat;
    bandwidth = bd_max;
    mexPrintf("bandwidth = %f\n", bandwidth);
    id_w0 = regspace(- ceil(bandwidth) + 1.0, ceil(bandwidth) - 1.0);
    weights0 = 1.0 - square(id_w0 / bandwidth);
    weights0 /= accu(weights0);
    
    
    for (i = 0; i < N; i ++) {
        idi = id_w0 + i;
        id_sub = find((idi >=0) % (idi < N));
        if (id_sub.n_elem < idi.n_elem) {
            w = weights0(id_sub);
            w /= accu(w);
            id_w = conv_to<uvec>::from(idi(id_sub));
        } else {
            w = weights0;
            id_w = conv_to<uvec>::from(idi);
        }
        
        S.slice(i) = (data.rows(id_w) % repmat(w, 1, P)).st() * conj(data.rows(id_w));
        //S.slice(i) = (S.slice(i) + S.slice(i).t()) / 2;
        /*std_vec = sqrt(real(S.slice(i).diag()));
        
        S_normalized = S.slice(i) / (std_vec * std_vec.t());
        Es_mat.row(i) = abs(S_normalized(idl)).t();*/
        S_normalized = S.slice(i);
        S_normalized.diag() += 1.0;
        K = inv_sympd(S_normalized);
        Es_mat.row(i) = abs(K(idl)).t();
    }
    
    Es_mat0 = Es_mat;
    cdf_s.fill(0.7); //(1.0 - (double) P / Pe);
    thr = as_scalar(quantile(vectorise(Es_mat), cdf_s));
    Es_mat.clean(thr); //(find(Es_mat < thr)).zeros();
    Es_mat(find(Es_mat >= thr)).fill(0.7); //ones(); //
    Es_mat.replace(0, 0.3);
    
    if (bandwidth < 2.0 * P) thr = 0.7;
    else {
        thr = 1.0 - pow(bandwidth / P / 2.0, 2);
        if (thr <= 0.7) thr = 0.7;
    }
    mexPrintf("chosen thr = %f\n", thr);
    mexEvalString("drawnow;");
    cdf_s.fill(thr); //(1.0 - (double) P / Pe);
    thr = as_scalar(quantile(vectorise(Es_mat0), cdf_s));
    Es_mat0.clean(thr);
    
    
    if (cdf_s(0) < 1.0 - (double) P / Pe) {
        K_cube.clear();
        
        for (i = 0; i < N; i ++) {
            id_nonzero = find(Es_mat0.row(i));
//                 mexPrintf("i = %i, id_nonzero = %i\n", i, id_nonzero.n_elem);
//                 mexEvalString("drawnow;");
            if (id_nonzero.n_elem > 0) {
                if (i > 0) {
                    K_init = diagmat(K.diag());
                    K_init(idl(id_nonzero)) = K(idl(id_nonzero));
                    K_init(idu(id_nonzero)) = K(idu(id_nonzero));
                } else {
                    S_normalized = S.slice(i);
                    //S_normalized *= bandwidth / P / 2.0;
                    S_normalized.diag() += 1.0; //(1 - bandwidth / P / 2.0);
                    S_normalized = inv_sympd(S_normalized);
                    
                    K_init = diagmat(S_normalized.diag());
                    K_init(idl(id_nonzero)) = S_normalized(idl(id_nonzero));
                    K_init(idu(id_nonzero)) = S_normalized(idu(id_nonzero));
                }
                K = GMRFParameterLearning(K_init, S.slice(i), idr(id_nonzero), idc(id_nonzero), 1000);
            } else K = diagmat(1 / diagvec(S.slice(i)));
            EKd_mat.row(i) = real(K.diag()).t();
            EJ_mat.row(i) = K(idl).st();
        }
        
    } else {
        
        for (i = 0; i < N; i ++) {
            
            id_nonzero = find(Es_mat0.row(i));
            //Rprintf("i = %i, n_nonzero = %i\n", i, id_nonzero.n_elem); //, min(eig_sym(S.slice(i))));
            if (id_nonzero.n_elem > 0) {
                id_zero = find(Es_mat0.row(i) - Es_old.row(i) < 0);
                if (id_zero.n_elem > 0) {
                    K_cube.slice(i)(idl(id_zero)).zeros();
                    K_cube.slice(i)(idu(id_zero)).zeros();
                }
                K = GMRFParameterLearning(K_cube.slice(i), S.slice(i), idr(id_nonzero), idc(id_nonzero), 1000);
            } else K = diagmat(1 / diagvec(S.slice(i)));
            
            EKd_mat.row(i) = real(K.diag()).t();
            EJ_mat.row(i) = K(idl).st();
        }
    }
    
    /*if (bandwidth >= 2.0 * P) {
        K = K_cube.slice(0);
        K_cube.clear();
        for (i = 0; i < N; i ++) {
//             mexPrintf("i = %i\n", i);
//             mexEvalString("drawnow;");
            if (S.slice(i).is_sympd()) K = inv_sympd(S.slice(i));
            else K = GMRFParameterLearning(K, S.slice(i), idr, idc, 1000);
            
            EKd_mat.row(i) = real(K.diag()).t();
            EJ_mat.row(i) = K(idl).st();
        }
    } else {
        thr = 1.0 - pow(bandwidth / P / 2.0, 2);
        if (thr <= 0.7) thr = 0.7;
        mexPrintf("chosen thr = %f\n", thr);
        mexEvalString("drawnow;");
        cdf_s.fill(thr); //(1.0 - (double) P / Pe);
        thr = as_scalar(quantile(vectorise(Es_mat0), cdf_s));
        Es_mat0.clean(thr);
        
        
        if (cdf_s(0) < 1.0 - (double) P / Pe) {
            K_cube.clear();
            
            for (i = 0; i < N; i ++) {
                id_nonzero = find(Es_mat0.row(i));
//                 mexPrintf("i = %i, id_nonzero = %i\n", i, id_nonzero.n_elem);
//                 mexEvalString("drawnow;");
                if (id_nonzero.n_elem > 0) {
                    if (i > 0) {
                        K_init = diagmat(K.diag());
                        K_init(idl(id_nonzero)) = K(idl(id_nonzero));
                        K_init(idu(id_nonzero)) = K(idu(id_nonzero));
                    } else {
                        S_normalized = S.slice(i);
                        S_normalized *= bandwidth / P / 2.0;
                        S_normalized.diag() += (1 - bandwidth / P / 2.0);
                        S_normalized = inv_sympd(S_normalized);
                        
                        K_init = diagmat(S_normalized.diag());
                        K_init(idl(id_nonzero)) = S_normalized(idl(id_nonzero));
                        K_init(idu(id_nonzero)) = S_normalized(idu(id_nonzero));
                    }
                    K = GMRFParameterLearning(K_init, S.slice(i), idr(id_nonzero), idc(id_nonzero), 1000);
                } else K = diagmat(1 / diagvec(S.slice(i)));
                EKd_mat.row(i) = real(K.diag()).t();
                EJ_mat.row(i) = K(idl).st();
            }
            
        } else {
            
            for (i = 0; i < N; i ++) {
                
                id_nonzero = find(Es_mat0.row(i));
                //Rprintf("i = %i, n_nonzero = %i\n", i, id_nonzero.n_elem); //, min(eig_sym(S.slice(i))));
                if (id_nonzero.n_elem > 0) {
                    id_zero = find(Es_mat0.row(i) - Es_old.row(i) < 0);
                    if (id_zero.n_elem > 0) {
                        K_cube.slice(i)(idl(id_zero)).zeros();
                        K_cube.slice(i)(idu(id_zero)).zeros();
                    }
                    K = GMRFParameterLearning(K_cube.slice(i), S.slice(i), idr(id_nonzero), idc(id_nonzero), 1000);
                } else K = diagmat(1 / diagvec(S.slice(i)));
                
                EKd_mat.row(i) = real(K.diag()).t();
                EJ_mat.row(i) = K(idl).st();
            }
        }


    }*/
    
    
    
    /*K = K_cube.slice(0);
    K_cube.clear();
    for (i = 0; i < N; i ++) {
        if (bandwidth < 2.0 * P) {
            S.slice(i) *= bandwidth / P / 2.0;
            S.slice(i).diag() += (1 - bandwidth / P / 2.0);  ///////////// need test
        }
        K = inv_sympd(S.slice(i));
        
        EKd_mat.row(i) = real(K.diag()).t();
        EJ_mat.row(i) = K(idl).st();
    }*/
    
    
    
    
    outputs(0) = cx_mat(Es_mat, zeros(N, Pe));
    outputs(1) = EJ_mat;
    outputs(2) = cx_mat(EKd_mat, zeros(N, P));
    
    return outputs;
}