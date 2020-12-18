#include "armaMex.hpp"
#include <boost/math/special_functions/digamma.hpp>
#include "CrossValidation.hpp"
#include "CrossValidation_QUIC.hpp"
#include "FwdBwdCHMM_var.hpp"
#include "SumEKodMDataUpdate.hpp"
#include "FwdBwdCHMM.hpp"
#include "FwdBwdDHMM.hpp"
#include "LogDetTriDiag.hpp"
#include "progressbar.hpp"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    // read data from MATLAB
    cx_mat data = armaGetCx(prhs[0]);
    int anneal_iters, max_iter;
    double T, tol_relative, tol_s;
    bool normalize_data;
    
    if (nrhs > 1) normalize_data = (bool)armaGetDouble(prhs[1]);
    else normalize_data = true;
    if (nrhs > 2) anneal_iters =(int)armaGetDouble(prhs[2]);
    else anneal_iters = 500;
    if (nrhs > 3) T = armaGetDouble(prhs[3]);
    else T = datum::inf;
    if (nrhs > 4) max_iter = (int)armaGetDouble(prhs[4]);
    else max_iter = 1e4;
    if (nrhs > 5) tol_relative = armaGetDouble(prhs[5]);
    else tol_relative = 1e-2;
    if (nrhs > 6) tol_s = armaGetDouble(prhs[6]);
    else tol_s = 1e-1;
    
    
    mexPrintf("BADGE starts...\n", anneal_iters);
    mexEvalString("drawnow;");
    // declare variables
    arma_rng::set_seed(0);
    auto t_start = std::chrono::high_resolution_clock::now();
    int P = data.n_cols, N = data.n_rows, Pe = P * (P - 1) / 2, Pa = Pe + P, i, j, iter;
    uvec idr(Pe), idc(Pe), idl, idu;
    uvec id_sub, id_w, id_rnd, idj, idr_vec(1), idc_vec(1), idj_vec(1), id_tmp, vP = regspace<uvec>(0, P - 1);
    umat idl_mat(P, P);
    double bandwidth_int = 3.0, bandwidth, c_init = 1.0, thr, a, b, c0, d0, c1, d1, Egamma = 1e6, Ealpha_rnd, bandwidth_ub, bandwidth_diff;
    double ELBO0, ELBOh, ELBO_max, ELBO_tmp, Hs_tmp, HJ_tmp, sum_EJ2_diff_tmp, eta_thr = 1.0, eta, eta_max, Hkappa_tmp, Hkappa_max, sum_Ekappa2_diff_tmp;
    bool is_pd;
    cx_mat data_rnd, S, K;
    mat Vdata, COVdata, Vdata_rnd, COVdata_rnd, w_mat, std_data, data2, data_rnd2;
    mat Es_mat(N, Pe), sum_pair_density(Pe, 4), Es_mat_old, log_B(N, 2, fill::zeros), sum_pair_density_tmp;
    vec Es_tmp, pL1pEs_true, cdf_s(1);
    mat22 log_A, log_A_rnd;
    rowvec2 log_s_init, log_s_rnd;
    cx_mat h_J, EJ_mat(N, Pe), E_diff;
    mat zeta_J_d(N, Pe), zeta_J_od(N - 1, Pe), VJ_mat, COVJ_mat, EJ2_mat;
    cx_vec h_J_tmp, h_J_j, EJ_tmp;
    vec zeta_J_d_tmp, zeta_J_od_tmp, zeta_J_d_j, zeta_J_od_j(N - 1), VJ_tmp, EJ2_tmp;
    cx_mat EKod_mat, EKod_mat_old, sum_EKod_m_data, sum_EKod_m_data_rnd, EKodi;
    mat EKod2_mat, VKod_mat, VKodi, Si;
    cx_vec pL1pEKod_true, pL1pEKod, EKod_tmp;
    vec pL1pVKodmn2_true, pL1pVKodmn2, EKod2_tmp;
    mat h_kappa, zeta_kappa_d(N, P), zeta_kappa_od(N - 1, P), Ekappa, Vkappa, COVkappa;
    mat pL1pEkappa, pL1pEkappa2mn2, natgrad_h_kappa, natgrad_zeta_kappa_d, natgrad_zeta_kappa_od;
    mat EKd_mat(N, P), EKdinv_mat, EKd_mat_old, pL1pEKdinv_true, pL1pEKdinv;
    vec h_kappa_tmp, zeta_kappa_d_tmp, zeta_kappa_od_tmp, Ekappa_tmp, Vkappa_tmp, EKd_tmp, EKdinv_tmp;
    vec weights0, w, std_vec, lb, ub, bd_range, eta_max_vec(P, fill::zeros), id_w0, idi;
    rowvec Ealpha(Pe), Hs(Pe), HJ, Hkappa, sum_EJ2_diff, sum_Ekappa2_diff, ELBO0_vec, Egamma_rnd;
    field<cx_mat> results_CHMM, results_CV;
    field<mat> results_DHMM, results_CHMM_var;
    logdet_result results_logdet;
    double diff_s, diff_d_max, diff_od_max, diff_d_r, diff_od_r;
    mat diff_d;
    cx_mat diff_od;
    
    
    // preprocessing
    
    j = 0;
    for (i = 0; i < P - 1; i ++) {
        idr(span(j, j + P - i - 2)) = regspace<uvec>(i + 1, P - 1);
        idc(span(j, j + P - i - 2)).fill(i);
        j = j + P - i - 1;
    }
    idl = idc * P + idr;
    idu = idr * P + idc;
    
    idl_mat(idl) = regspace<uvec>(0, Pe - 1);
    idl_mat(idu) = idl_mat(idl);
    idl_mat.diag().zeros(); //regspace<uvec>(Pe, Pa - 1);
    
    
    
    
    if (P >= N) {
        VKodi.set_size(P, P);
        VKodi.zeros();
    }
    
    // normalize data using estimated time-varying mean and variance
    if (normalize_data) {
//         data.each_row() -= mean(data);
//         data.each_row() /= cx_rowvec(stddev(data), rowvec(P, fill::zeros)); // / sqrt(2.0)
        bandwidth = CrossValidation(data, N, N, P);
        mexPrintf("bandwidth = %f.\n", bandwidth);
        mexEvalString("drawnow;");
        id_w0 = regspace(- ceil(bandwidth) + 1.0, ceil(bandwidth) - 1.0);
        weights0 = 1.0 - square(id_w0 / bandwidth);
        weights0 /= accu(weights0);
        std_data.set_size(N, P);
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
            
            std_data.row(i) = sqrt(sum(real(data.rows(id_w) % conj(data.rows(id_w))) %
                    repmat(w, 1, P)));
        }
        data /= cx_mat(std_data / sqrt(2.0), mat(N, P, fill::zeros));  // 
        std_data.clear();
    }
    data2 = real(data % conj(data));
    // initialize other variables
    results_CV = CrossValidation_QUIC(data, idr, idc, idl, idu, N, P, Pe);
    Es_mat = real(results_CV(0));
    EJ_mat = results_CV(1);
    EKd_mat = real(results_CV(2));
    results_CV.clear();

    
    // initialize s
    
    
    a = 1.0 + Pe - accu(Es_mat.row(0));
    b = 1.0 + accu(Es_mat.row(0));
    log_s_init(0) = boost::math::digamma(a) - boost::math::digamma(a + b);
    log_s_init(1) = boost::math::digamma(b) - boost::math::digamma(a + b);
    
    sum_pair_density.col(0) = conv_to<vec>::from(sum((Es_mat.head_rows(N - 1) < 0.5) % (Es_mat.tail_rows(N - 1) < 0.5)));
    sum_pair_density.col(1) = conv_to<vec>::from(sum((Es_mat.head_rows(N - 1) < 0.5) % (Es_mat.tail_rows(N - 1) > 0.5)));
    sum_pair_density.col(2) = conv_to<vec>::from(sum((Es_mat.head_rows(N - 1) > 0.5) % (Es_mat.tail_rows(N - 1) < 0.5)));
    sum_pair_density.col(3) = conv_to<vec>::from(sum((Es_mat.head_rows(N - 1) > 0.5) % (Es_mat.tail_rows(N - 1) > 0.5)));
    
    
    c0 = 1.0 + accu(sum_pair_density.col(0));
    d0 = 1.0 + accu(sum_pair_density.col(1));
    c1 = 1.0 + accu(sum_pair_density.col(3));
    d1 = 1.0 + accu(sum_pair_density.col(2));
    
    log_A(0, 0) = boost::math::digamma(c0) - boost::math::digamma(c0 + d0);
    log_A(0, 1) = boost::math::digamma(d0) - boost::math::digamma(c0 + d0);
    log_A(1, 1) = boost::math::digamma(c1) - boost::math::digamma(c1 + d1);
    log_A(1, 0) = boost::math::digamma(d1) - boost::math::digamma(c1 + d1);
    
    for (j = 0; j < Pe; j ++) {
        results_DHMM = FwdBwdDHMM(log_s_init, log_A, join_rows(1 - Es_mat.col(j), Es_mat.col(j)), N, 2);
        Es_mat.col(j) = results_DHMM(0);
        sum_pair_density_tmp = results_DHMM(1);
        sum_pair_density.row(j) = join_rows(sum_pair_density_tmp.row(0), sum_pair_density_tmp.row(1));
        Hs(j) = as_scalar(results_DHMM(2));
    }

    
    // initialize J
    E_diff = EJ_mat.head_rows(N - 1) - EJ_mat.tail_rows(N - 1);
    Ealpha = (N - 1.0) / (sum(real(E_diff % conj(E_diff))) + 1e-10);//.fill(1e6);
    E_diff.clear();
    zeta_J_d.fill(c_init);
    zeta_J_d.row(0) += Ealpha;
    zeta_J_d.row(N - 1) += Ealpha;
    zeta_J_d.rows(1, N - 2).each_row() += 2 * Ealpha;
    zeta_J_od.each_row() = Ealpha;
    h_J = join_cols(EJ_mat.row(0) - EJ_mat.row(1), 2 * EJ_mat.rows(span(1, N - 2)) - EJ_mat.rows(span(0, N - 3)) - EJ_mat.rows(span(2, N - 1)),
            EJ_mat.row(N - 1) - EJ_mat.row(N - 2));
    h_J.each_row() %= cx_rowvec(Ealpha, rowvec(Pe, fill::zeros));
    h_J += c_init * EJ_mat;
    results_CHMM_var = FwdBwdCHMM_var(zeta_J_d, zeta_J_od, N, Pe);
    VJ_mat = results_CHMM_var(0);
    COVJ_mat = results_CHMM_var(1);
    EJ2_mat = real(EJ_mat % conj(EJ_mat)) + VJ_mat;
    sum_EJ2_diff = sum(square(abs(EJ_mat.head_rows(N - 1) - EJ_mat.tail_rows(N - 1)))) + results_CHMM_var(2);
    
    results_logdet = LogDetTriDiag(zeta_J_d, zeta_J_od, N, Pe);
    HJ = - results_logdet.logdetK;
    
    EKod_mat = EJ_mat % Es_mat;
    EKod2_mat = EJ2_mat % Es_mat;
    // if (is_rnd_missing) VKod_mat = EKod2_mat - square(EKod_mat);
    sum_EKod_m_data = SumEKodMDataUpdate(data, EKod_mat, idl, idu, idl_mat, N, P);
    
    // initialize kappa
    Ekappa = log(EKd_mat);
    Egamma = (N - 1.0) * P / accu(square(Ekappa.head_rows(N - 1) - Ekappa.tail_rows(N - 1)));
    zeta_kappa_d.fill(c_init);
    zeta_kappa_d.row(0) += Egamma;
    zeta_kappa_d.row(N - 1) += Egamma;
    zeta_kappa_d.rows(1, N - 2) += 2 * Egamma;
    zeta_kappa_od.fill(Egamma);
    h_kappa = Egamma * join_cols(Ekappa.row(0) - Ekappa.row(1), 2 * Ekappa.rows(span(1, N - 2)) - Ekappa.rows(span(0, N - 3)) - Ekappa.rows(span(2, N - 1)),
            Ekappa.row(N - 1) - Ekappa.row(N - 2)) + c_init * Ekappa;
    results_CHMM_var = FwdBwdCHMM_var(zeta_kappa_d, zeta_kappa_od, N, P);
    Vkappa = results_CHMM_var(0);
    COVkappa = results_CHMM_var(1);
    sum_Ekappa2_diff = sum(square(Ekappa.head_rows(N - 1) - Ekappa.tail_rows(N - 1))) + results_CHMM_var(2);
    results_logdet = LogDetTriDiag(zeta_kappa_d, zeta_kappa_od, N, P);
    Hkappa = - results_logdet.logdetK / 2;
    
    EKd_mat = exp(Ekappa + Vkappa / 2);
    EKdinv_mat = exp(- Ekappa + Vkappa / 2);
    
    EKd_mat_old = EKd_mat;
    Es_mat_old = Es_mat;
    EKod_mat_old = EKod_mat;
    // initialize parameters for simulated annealing
    bandwidth_ub = (N / 2.0 - 1.0) * (1.0 - 1.0 / T) + 1.0;
    bandwidth = bandwidth_ub;
    
    lb = regspace(0, N - 1) - ceil(bandwidth) + 0.5;
    lb(find(lb < -0.5)).fill(-0.5);
    ub = regspace(0, N - 1) + ceil(bandwidth) - 0.5;
    ub(find(ub > N - 0.5)).fill(N - 0.5);
    bd_range = ub - lb;
    bandwidth_diff = (bandwidth_ub - 1.0) / anneal_iters * 10.0;
    
    mexPrintf("Start simulated annealing with %i iterations...\n", anneal_iters);
    mexEvalString("drawnow;");
    progressbar pb(anneal_iters);
   
    // Natural Gradient Variational Inference
    for (iter = 1; iter <= max_iter; iter ++) {
        
        if (iter > 1 && T > 1) {
            id_rnd = conv_to<uvec>::from(round(lb + bd_range % randu(N)));
            
            data_rnd = data.rows(id_rnd);
            data_rnd2 = real(data_rnd % conj(data_rnd));
            sum_EKod_m_data_rnd = SumEKodMDataUpdate(data_rnd, EKod_mat, idl, idu, idl_mat, N, P);
            
            
            idj = randperm(Pe);
            
            for (i = 0; i < Pe; i ++){
                j = idj(i);
                
                // compute gradient wrt off-diagonal elements in K
                
                sum_EKod_m_data.col(idr(j)) -= EKod_mat.col(j) % data.col(idc(j));
                sum_EKod_m_data.col(idc(j)) -= conj(EKod_mat.col(j)) % data.col(idr(j));
                
                
                pL1pEKod_true = - EKdinv_mat.col(idr(j)) % data.col(idc(j)) % conj(sum_EKod_m_data.col(idr(j))) -
                        EKdinv_mat.col(idc(j)) % conj(data.col(idr(j))) % sum_EKod_m_data.col(idc(j)) -
                        2 * data.col(idc(j)) % conj(data.col(idr(j)));
                pL1pVKodmn2_true = EKdinv_mat.col(idr(j)) % data2.col(idc(j)) +
                        EKdinv_mat.col(idc(j)) % data2.col(idr(j));
                
                
                sum_EKod_m_data_rnd.col(idr(j)) -= EKod_mat.col(j) % data_rnd.col(idc(j));
                sum_EKod_m_data_rnd.col(idc(j)) -= conj(EKod_mat.col(j)) % data_rnd.col(idr(j));
                
                pL1pEKod = - EKdinv_mat.col(idr(j)) % data_rnd.col(idc(j)) % conj(sum_EKod_m_data_rnd.col(idr(j))) -
                        EKdinv_mat.col(idc(j)) % conj(data_rnd.col(idr(j))) % sum_EKod_m_data_rnd.col(idc(j)) -
                        2 * data_rnd.col(idc(j)) % conj(data_rnd.col(idr(j)));
                pL1pVKodmn2 = EKdinv_mat.col(idr(j)) % data_rnd2.col(idc(j)) +
                        EKdinv_mat.col(idc(j)) % data_rnd2.col(idr(j));
                
                // update s
                pL1pEs_true = 2 * real(pL1pEKod_true % EJ_mat.col(j)) - pL1pVKodmn2_true % EJ2_mat.col(j);
                ELBO0 = log_s_init(0) * (1 - Es_mat(0, j)) + log_s_init(1) * Es_mat(0, j) + accu(pL1pEs_true % Es_mat.col(j)) +
                        accu(sum_pair_density.row(j) % join_rows(log_A.row(0), log_A.row(1))) + Hs(j);
                
                log_s_rnd(0) = randg(distr_param(a, 1.0));
                log_s_rnd(1) = randg(distr_param(b, 1.0));
                log_s_rnd = log(log_s_rnd) - log(accu(log_s_rnd));
                log_s_rnd *= (1.0 - 1.0 / T);
                log_s_rnd += log_s_init / T;
                
                log_A_rnd(0, 0) = randg(distr_param(c0, 1.0));
                log_A_rnd(0, 1) = randg(distr_param(d0, 1.0));
                log_A_rnd(1, 0) = randg(distr_param(d1, 1.0));
                log_A_rnd(1, 1) = randg(distr_param(c1, 1.0));
                log_A_rnd = mat(log(log_A_rnd)).each_col() -  log(sum(log_A_rnd, 1)); //- repmat(log(sum(log_A_rnd, 1)), 1, 2);
                log_A_rnd *= (1.0 - 1.0 / T);
                log_A_rnd += log_A / T;
                
                log_B.col(1) = 2 * real(pL1pEKod % EJ_mat.col(j)) - pL1pVKodmn2 % EJ2_mat.col(j);
                
                results_DHMM = FwdBwdDHMM(log_s_rnd, log_A_rnd, log_B, N, 2);   //log_A
                Es_tmp = results_DHMM(0);
                sum_pair_density_tmp = results_DHMM(1);
                Hs_tmp = as_scalar(results_DHMM(2));
                ELBOh = log_s_init(0) * (1 - Es_tmp(0)) + log_s_init(1) * Es_tmp(0) +
                        accu(pL1pEs_true % Es_tmp) + accu(sum_pair_density_tmp % log_A) + Hs_tmp;
                
                if (randu() < exp((ELBOh - ELBO0) / (1 - 1 / T) / 2)) { //(ELBOh > ELBO0) { //
                    Es_mat.col(j) = Es_tmp;
                    sum_pair_density.row(j) = join_rows(sum_pair_density_tmp.row(0), sum_pair_density_tmp.row(1));
                    Hs(j) = Hs_tmp;
                    EKod_mat.col(j) = Es_mat.col(j) % EJ_mat.col(j);
                    EKod2_mat.col(j) = Es_mat.col(j) % EJ2_mat.col(j);
                }
                
                // update J
                
                ELBO0 = 2 * accu(real(pL1pEKod_true % EKod_mat.col(j))) - accu(pL1pVKodmn2_true % EKod2_mat.col(j)) -
                        Ealpha(j) * sum_EJ2_diff(j) + HJ(j);
                
                
                zeta_J_d_j = pL1pVKodmn2_true % Es_mat.col(j);
                zeta_J_d_j(0) += Ealpha(j);
                zeta_J_d_j(N - 1) += Ealpha(j);
                zeta_J_d_j(span(1, N - 2)) += 2 * Ealpha(j);
                zeta_J_od_j.fill(Ealpha(j));
                is_pd = (bool)as_scalar(FwdBwdCHMM_var(zeta_J_d_j, zeta_J_od_j, N, 1)(3));
                
                if (is_pd) eta_max = eta_thr;
                else {
                    h_J_j = conj(pL1pEKod_true) % Es_mat.col(j);
                    ELBO_max = ELBO0;
                    eta_max = 0;
                    eta = eta_thr / 2;
                    while (eta > 1e-2) {
                        h_J_tmp = (1 - eta) * h_J.col(j) + eta * h_J_j;
                        zeta_J_d_tmp = (1 - eta) * zeta_J_d.col(j) + eta * zeta_J_d_j;
                        zeta_J_od_tmp = (1 - eta) * zeta_J_od.col(j) + eta * zeta_J_od_j;
                        results_CHMM = FwdBwdCHMM(h_J_tmp, zeta_J_d_tmp, zeta_J_od_tmp, N, 1);
                        is_pd = (bool)as_scalar(real(results_CHMM(4)));
                        if (is_pd) {
                            results_logdet = LogDetTriDiag(zeta_J_d_tmp, zeta_J_od_tmp, N, 1);
                            HJ_tmp = - as_scalar(results_logdet.logdetK);
                            
                            EJ_tmp = results_CHMM(0);
                            VJ_tmp = real(results_CHMM(1));
                            sum_EJ2_diff_tmp = as_scalar(real(results_CHMM(3)));
                            
                            EJ2_tmp = real(EJ_tmp % conj(EJ_tmp)) + VJ_tmp;
                            EKod_tmp = Es_mat.col(j) % EJ_tmp;
                            EKod2_tmp = Es_mat.col(j) % EJ2_tmp;
                            
                            ELBOh = 2 * accu(real(pL1pEKod_true % EKod_tmp)) - accu(pL1pVKodmn2_true % EKod2_tmp) -
                                    Ealpha(j) * sum_EJ2_diff_tmp + HJ_tmp;
                            if (ELBO_max > ELBO0 && ELBOh <= ELBO_max) break;
                            else {
                                if (ELBOh > ELBO_max) {
                                    eta_max = eta;
                                    ELBO_max = ELBOh;
                                }
                                eta /= 2;
                            }
                        } else eta /= 2;
                    }
                }
                
                if (eta_max > 0) {
                    Ealpha_rnd = Ealpha(j) / T + randg(distr_param((N - 1.0), 1.0 / sum_EJ2_diff(j))) * (1.0 - 1.0 / T);
                    
                    zeta_J_d_j = pL1pVKodmn2 % Es_mat.col(j);
                    zeta_J_d_j(0) += Ealpha_rnd;
                    zeta_J_d_j(N - 1) += Ealpha_rnd;
                    zeta_J_d_j(span(1, N - 2)) += 2 * Ealpha_rnd;
                    zeta_J_od_j.fill(Ealpha_rnd);
                    h_J_tmp = (1 - eta_max) * h_J.col(j) + eta_max * conj(pL1pEKod) % Es_mat.col(j);
                    zeta_J_d_tmp = (1 - eta_max) * zeta_J_d.col(j) + eta_max * zeta_J_d_j;
                    zeta_J_od_tmp = (1 - eta_max) * zeta_J_od.col(j) + eta_max * zeta_J_od_j;
                    
                    results_CHMM = FwdBwdCHMM(h_J_tmp, zeta_J_d_tmp, zeta_J_od_tmp, N, 1);
                    is_pd = (bool)as_scalar(real(results_CHMM(4)));
                    if (is_pd) {
                        results_logdet = LogDetTriDiag(zeta_J_d_tmp, zeta_J_od_tmp, N, 1);
                        HJ_tmp = - as_scalar(results_logdet.logdetK);
                        
                        EJ_tmp = results_CHMM(0);
                        VJ_tmp = real(results_CHMM(1));
                        sum_EJ2_diff_tmp = as_scalar(real(results_CHMM(3)));
                        
                        EJ2_tmp = real(EJ_tmp % conj(EJ_tmp)) + VJ_tmp;
                        EKod_tmp = Es_mat.col(j) % EJ_tmp;
                        EKod2_tmp = Es_mat.col(j) % EJ2_tmp;
                        
                        ELBOh = 2 * accu(real(pL1pEKod_true % EKod_tmp)) - accu(pL1pVKodmn2_true % EKod2_tmp) -
                                Ealpha(j) * sum_EJ2_diff_tmp + HJ_tmp;
                        if (randu() < exp((ELBOh - ELBO0) / (1 - 1 / T) / 2)) { //(ELBOh > ELBO0) {//
                            h_J.col(j) = h_J_tmp;
                            zeta_J_d.col(j) = zeta_J_d_tmp;
                            zeta_J_od.col(j) = zeta_J_od_tmp;
                            EJ_mat.col(j) = EJ_tmp;
                            EJ2_mat.col(j) = EJ2_tmp;
                            HJ(j) = HJ_tmp;
                            sum_EJ2_diff(j) = sum_EJ2_diff_tmp;
                            Ealpha(j) = (N - 1.0) / sum_EJ2_diff_tmp;
                            
                            
                            EKod_mat.col(j) = EKod_tmp;
                            EKod2_mat.col(j) = EKod2_tmp;
                            
                        }
                    }
                }
                
                sum_EKod_m_data.col(idr(j)) += EKod_mat.col(j) % data.col(idc(j));
                sum_EKod_m_data.col(idc(j)) += conj(EKod_mat.col(j)) % data.col(idr(j));
                sum_EKod_m_data_rnd.col(idr(j)) += EKod_mat.col(j) % data_rnd.col(idc(j));
                sum_EKod_m_data_rnd.col(idc(j)) += conj(EKod_mat.col(j)) % data_rnd.col(idr(j));
            }
            
            VKod_mat = EKod2_mat - real(EKod_mat % conj(EKod_mat));
            
            // compute gradient wrt diagonal elements in K
            
            pL1pEKdinv_true = - real(sum_EKod_m_data % conj(sum_EKod_m_data));
            pL1pEKdinv = - real(sum_EKod_m_data_rnd % conj(sum_EKod_m_data_rnd));
            
            
            if (P < N) {
                for (j = 0; j < P; j ++) {
                    id_tmp = join_cols(vP.head(j), vP.tail(P - 1 - j));
                    idj_vec.fill(j);
                    pL1pEKdinv.col(j) -= sum(data_rnd2.cols(id_tmp) % VKod_mat.cols(idl_mat(idj_vec, id_tmp)), 1);
                    pL1pEKdinv_true.col(j) -= sum(data2.cols(id_tmp) % VKod_mat.cols(idl_mat(idj_vec, id_tmp)), 1);
                }
            } else {
                for (i = 0; i < N; i ++) {
                    VKodi(idl) = VKod_mat.row(i);
                    VKodi(idu) = VKod_mat.row(i);
                    
                    pL1pEKdinv.row(i) -= data_rnd2.row(i) * VKodi;   //sum(EKodi * nSi % EKodi, 1).t()
                    pL1pEKdinv_true.row(i) -= data2.row(i) * VKodi;
                }
            }
            
            
            
            pL1pEkappa = 1 - data2 % EKd_mat % (1 - Ekappa) -
                        pL1pEKdinv_true % EKdinv_mat % (1 + Ekappa);
            pL1pEkappa2mn2 = data2 % EKd_mat - pL1pEKdinv_true % EKdinv_mat;
            ELBO0_vec = sum(Ekappa) - sum(pL1pEkappa2mn2) - Egamma / 2 * sum_Ekappa2_diff + Hkappa;
            
            // update kappa
            natgrad_h_kappa = pL1pEkappa - h_kappa;
            natgrad_zeta_kappa_d = pL1pEkappa2mn2 - zeta_kappa_d;
            natgrad_zeta_kappa_d.row(0) += Egamma;
            natgrad_zeta_kappa_d.row(N - 1) += Egamma;
            natgrad_zeta_kappa_d.rows(1, N - 2) += 2 * Egamma;
            natgrad_zeta_kappa_od = Egamma - zeta_kappa_od;
            
            
            
            for (j = 0; j < P; j ++) {
                ELBO_max = ELBO0_vec(j);
                eta = eta_thr;
                while (eta > 1e-10) {
                    h_kappa_tmp = h_kappa.col(j) + eta * natgrad_h_kappa.col(j);
                    zeta_kappa_d_tmp = zeta_kappa_d.col(j) + eta * natgrad_zeta_kappa_d.col(j);
                    zeta_kappa_od_tmp = zeta_kappa_od.col(j) + eta * natgrad_zeta_kappa_od.col(j);
                    results_logdet = LogDetTriDiag(zeta_kappa_d_tmp, zeta_kappa_od_tmp, N, 1);
                    if (results_logdet.is_pd) {
                        results_CHMM = FwdBwdCHMM(cx_mat(h_kappa_tmp, mat(N, 1, fill::zeros)), 
                                zeta_kappa_d_tmp, zeta_kappa_od_tmp, N, 1);
                        Ekappa_tmp = real(results_CHMM(0));
                        Vkappa_tmp = real(results_CHMM(1));
                        sum_Ekappa2_diff_tmp = as_scalar(real(results_CHMM(3)));
                        EKd_tmp = exp(Ekappa_tmp + Vkappa_tmp / 2);
                        EKdinv_tmp = exp(- Ekappa_tmp + Vkappa_tmp / 2);
                        ELBO_tmp = accu(Ekappa_tmp) - accu(data2.col(j) % EKd_tmp) +
                                accu(pL1pEKdinv_true.col(j) % EKdinv_tmp)- Egamma / 2 *
                                sum_Ekappa2_diff_tmp - results_logdet.logdetK(0) / 2;
                        if (ELBO_max > ELBO0_vec(j) && ELBO_tmp <= ELBO_max) {
                            break;
                        } else {
                            if (ELBO_tmp > ELBO_max) {
                                eta_max_vec(j) = eta;
                                ELBO_max = ELBO_tmp;
                            }
                            eta /= 2;
                        }
                    }
                    else
                        eta /= 2;
                }
            }
            
            pL1pEkappa = 1 - data_rnd2 % EKd_mat % (1 - Ekappa) -
                    pL1pEKdinv % EKdinv_mat % (1 + Ekappa);
            pL1pEkappa2mn2 = data_rnd2 % EKd_mat - pL1pEKdinv % EKdinv_mat;
            
            Egamma_rnd = Egamma / T + (1.0 - 1.0 / T) * randg<rowvec>(P, distr_param((N - 1.0) * P / 2.0, 2.0 / accu(sum_Ekappa2_diff)));
            
            natgrad_h_kappa = pL1pEkappa - h_kappa;
            natgrad_zeta_kappa_d = pL1pEkappa2mn2 - zeta_kappa_d;
            natgrad_zeta_kappa_d.row(0) += Egamma_rnd; //Egamma; //
            natgrad_zeta_kappa_d.row(N - 1) += Egamma_rnd; //Egamma;  //
            natgrad_zeta_kappa_d.rows(1, N - 2).each_row() += 2 * Egamma_rnd; //repmat(2 * Egamma_rnd, N - 2, 1); // 2 * Egamma; //
            natgrad_zeta_kappa_od = - zeta_kappa_od;
            natgrad_zeta_kappa_od.each_row() += Egamma_rnd;
            
            for (j = 0; j < P; j ++) {
                h_kappa_tmp = h_kappa.col(j) + eta_max_vec(j) * natgrad_h_kappa.col(j);
                zeta_kappa_d_tmp = zeta_kappa_d.col(j) + eta_max_vec(j) * natgrad_zeta_kappa_d.col(j);
                zeta_kappa_od_tmp = zeta_kappa_od.col(j) + eta_max_vec(j) * natgrad_zeta_kappa_od.col(j);
                results_logdet = LogDetTriDiag(zeta_kappa_d_tmp, zeta_kappa_od_tmp, N, 1);
                if (results_logdet.is_pd) {
                    results_CHMM = FwdBwdCHMM(cx_mat(h_kappa_tmp, mat(N, 1, fill::zeros)),
                            zeta_kappa_d_tmp, zeta_kappa_od_tmp, N, 1);
                    Ekappa_tmp = real(results_CHMM(0));
                    Vkappa_tmp = real(results_CHMM(1));
                    sum_Ekappa2_diff_tmp = as_scalar(real(results_CHMM(3)));
                    EKd_tmp = exp(Ekappa_tmp + Vkappa_tmp / 2);
                    EKdinv_tmp = exp(- Ekappa_tmp + Vkappa_tmp / 2);
                    Hkappa_tmp = - results_logdet.logdetK(0) / 2;
                    ELBOh = accu(Ekappa_tmp) - accu(data2.col(j) % EKd_tmp) +
                            accu(pL1pEKdinv_true.col(j) % EKdinv_tmp) -
                            Egamma / 2 * sum_Ekappa2_diff_tmp + Hkappa_tmp;
                    if (randu() < exp((ELBOh - ELBO0_vec(j)) / (1 - 1 / T) / 2)) { //(ELBOh > ELBO0_vec(j)) { //
                        h_kappa.col(j) =  h_kappa_tmp;
                        zeta_kappa_d.col(j) = zeta_kappa_d_tmp;
                        zeta_kappa_od.col(j) = zeta_kappa_od_tmp;
                        Ekappa.col(j) = Ekappa_tmp;
                        sum_Ekappa2_diff(j) = sum_Ekappa2_diff_tmp;
                        EKd_mat.col(j) = EKd_tmp;
                        EKdinv_mat.col(j) = EKdinv_tmp;
                        Hkappa(j) = Hkappa_tmp;
                    }
                }
            }
            
        } else {
            
            
            idj = randperm(Pe);
            for (i = 0; i < Pe; i ++){
                j = idj(i);
                
                sum_EKod_m_data.col(idr(j)) -= EKod_mat.col(j) % data.col(idc(j));
                sum_EKod_m_data.col(idc(j)) -= conj(EKod_mat.col(j)) % data.col(idr(j));
                
                
                pL1pEKod_true = - EKdinv_mat.col(idr(j)) % data.col(idc(j)) % conj(sum_EKod_m_data.col(idr(j))) -
                        EKdinv_mat.col(idc(j)) % conj(data.col(idr(j))) % sum_EKod_m_data.col(idc(j)) -
                        2 * data.col(idc(j)) % conj(data.col(idr(j)));
                pL1pVKodmn2_true = EKdinv_mat.col(idr(j)) % data2.col(idc(j)) +
                        EKdinv_mat.col(idc(j)) % data2.col(idr(j));
                
                
                // update s
                log_B.col(1) = 2 * real(pL1pEKod_true % EJ_mat.col(j)) - pL1pVKodmn2_true % EJ2_mat.col(j);
                
                results_DHMM = FwdBwdDHMM(log_s_init, log_A, log_B, N, 2);
                Es_mat.col(j) = results_DHMM(0);
                sum_pair_density.row(j) = join_rows(results_DHMM(1).row(0),results_DHMM(1).row(1));
                if (iter == 1) Hs(j) = as_scalar(results_DHMM(2));
                
                EKod_mat.col(j) = Es_mat.col(j) % EJ_mat.col(j);
                EKod2_mat.col(j) = Es_mat.col(j) % EJ2_mat.col(j);
                
                
                // update J
                ELBO0 = 2 * accu(real(pL1pEKod_true % EKod_mat.col(j))) - accu(pL1pVKodmn2_true % EKod2_mat.col(j))
                - Ealpha(j) * sum_EJ2_diff(j) + HJ(j);
        
                h_J_j = conj(pL1pEKod_true) % Es_mat.col(j);
                zeta_J_d_j = pL1pVKodmn2_true % Es_mat.col(j);
                zeta_J_d_j(0) += Ealpha(j);
                zeta_J_d_j(N - 1) += Ealpha(j);
                zeta_J_d_j(span(1, N - 2)) += 2 * Ealpha(j);
                zeta_J_od_j.fill(Ealpha(j));
                is_pd = (bool)FwdBwdCHMM_var(zeta_J_d_j, zeta_J_od_j, N, 1)(3)(0);
                
                if (is_pd) eta_max = eta_thr;
                else {
                    ELBO_max = ELBO0;
                    eta_max = 0;
                    eta = eta_thr / 2;
                    while (eta > 1e-4) {
                        h_J_tmp = (1 - eta) * h_J.col(j) + eta * h_J_j;
                        zeta_J_d_tmp = (1 - eta) * zeta_J_d.col(j) + eta * zeta_J_d_j;
                        zeta_J_od_tmp = (1 - eta) * zeta_J_od.col(j) + eta * zeta_J_od_j;
                        results_CHMM = FwdBwdCHMM(h_J_tmp, zeta_J_d_tmp, zeta_J_od_tmp, N, 1);
                        is_pd = (bool)as_scalar(real(results_CHMM(4)));
                        if (is_pd) {
                            results_logdet = LogDetTriDiag(zeta_J_d_tmp, zeta_J_od_tmp, N, 1);
                            HJ_tmp = - as_scalar(results_logdet.logdetK);
                            
                            EJ_tmp = results_CHMM(0);
                            VJ_tmp = real(results_CHMM(1));
                            sum_EJ2_diff_tmp = as_scalar(real(results_CHMM(3)));
                            
                            EJ2_tmp = real(EJ_tmp % conj(EJ_tmp)) + VJ_tmp;
                            EKod_tmp = Es_mat.col(j) % EJ_tmp;
                            EKod2_tmp = Es_mat.col(j) % EJ2_tmp;
                            
                            ELBOh = 2 * accu(real(pL1pEKod_true % EKod_tmp)) - accu(pL1pVKodmn2_true % EKod2_tmp) -
                                    Ealpha(j) * sum_EJ2_diff_tmp + HJ_tmp;
                            if (ELBO_max > ELBO0 && ELBOh <= ELBO_max) break;
                            else {
                                if (ELBOh > ELBO_max) {
                                    eta_max = eta;
                                    ELBO_max = ELBOh;
                                }
                                eta /= 2;
                            }
                        } else eta /= 2;
                    }
                }
                
                if (eta_max > 0) {
                    h_J.col(j) = (1 - eta_max) * h_J.col(j) + eta_max * h_J_j;
                    zeta_J_d.col(j) = (1 - eta_max) * zeta_J_d.col(j) + eta_max * zeta_J_d_j;
                    zeta_J_od.col(j) = (1 - eta_max) * zeta_J_od.col(j) + eta_max * zeta_J_od_j;
                    
                    results_CHMM = FwdBwdCHMM(h_J.col(j), zeta_J_d.col(j), zeta_J_od.col(j), N, 1);
                    EJ_mat.col(j) = results_CHMM(0);
                    VJ_mat.col(j) = real(results_CHMM(1));
                    COVJ_mat.col(j) = real(results_CHMM(2));
                    //sum_EJ2_diff_tmp = as<double>(results_CHMM["sum_E2_diff"]);
                    sum_EJ2_diff(j) = as_scalar(real(results_CHMM(3)));
                    Ealpha(j) = (N - 1.0) / sum_EJ2_diff(j);
                    
                    EJ2_mat.col(j) = real(EJ_mat.col(j) % conj(EJ_mat.col(j))) + VJ_mat.col(j);
                    EKod_mat.col(j) = Es_mat.col(j) % EJ_mat.col(j);
                    EKod2_mat.col(j) = Es_mat.col(j) % EJ2_mat.col(j);
                    
                    
                    results_logdet = LogDetTriDiag(zeta_J_d.col(j), zeta_J_od.col(j), N, 1);
                    HJ(j) = - as_scalar(results_logdet.logdetK);
                    
                }
                
                sum_EKod_m_data.col(idr(j)) += EKod_mat.col(j) % data.col(idc(j));
                sum_EKod_m_data.col(idc(j)) += conj(EKod_mat.col(j)) % data.col(idr(j));
                
            }
            
            VKod_mat = EKod2_mat - real(EKod_mat % conj(EKod_mat));
            
            // Update Diagonal elements in K
            
            pL1pEKdinv_true = - real(sum_EKod_m_data % conj(sum_EKod_m_data));
            
            if (P < N) {
                for (j = 0; j < P; j ++) {
                    id_tmp = join_cols(vP.head(j), vP.tail(P - 1 - j));
                    idj_vec.fill(j);
                    pL1pEKdinv_true.col(j) -= sum(data2.cols(id_tmp) % VKod_mat.cols(idl_mat(idj_vec, id_tmp)), 1);
                }
            } else {
                for (i = 0; i < N; i ++) {
                    VKodi(idl) = VKod_mat.row(i);
                    VKodi(idu) = VKod_mat.row(i);
                    
                    pL1pEKdinv_true.row(i) -= data2.row(i) * VKodi;
                }
            }
            
            
            
            
            pL1pEkappa = 1 - data2 % EKd_mat % (1 - Ekappa) -
                    pL1pEKdinv_true % EKdinv_mat % (1 + Ekappa);
            pL1pEkappa2mn2 = data2 % EKd_mat - pL1pEKdinv_true % EKdinv_mat;
            
            ELBO0_vec = sum(Ekappa) - sum(pL1pEkappa2mn2) - Egamma / 2 * sum_Ekappa2_diff + Hkappa;
            
            natgrad_h_kappa = pL1pEkappa - h_kappa;
            natgrad_zeta_kappa_d = pL1pEkappa2mn2 - zeta_kappa_d;
            natgrad_zeta_kappa_d.row(0) += Egamma;
            natgrad_zeta_kappa_d.row(N - 1) += Egamma;
            natgrad_zeta_kappa_d.rows(1, N - 2) += 2 * Egamma;
            natgrad_zeta_kappa_od = Egamma - zeta_kappa_od;
            
            for (j = 0; j < P; j ++) {
                ELBO_max = ELBO0_vec(j);
                Hkappa_max = Hkappa(j);
                eta = eta_thr;
                while (eta > 1e-10) {
                    h_kappa_tmp = h_kappa.col(j) + eta * natgrad_h_kappa.col(j);
                    zeta_kappa_d_tmp = zeta_kappa_d.col(j) + eta * natgrad_zeta_kappa_d.col(j);
                    zeta_kappa_od_tmp = zeta_kappa_od.col(j) + eta * natgrad_zeta_kappa_od.col(j);
                    results_logdet = LogDetTriDiag(zeta_kappa_d_tmp, zeta_kappa_od_tmp, N, 1);
                    if (results_logdet.is_pd) {
                        results_CHMM = FwdBwdCHMM(cx_mat(h_kappa_tmp, mat(N, 1, fill::zeros)),
                                zeta_kappa_d_tmp, zeta_kappa_od_tmp, N, 1);
                        Ekappa_tmp = real(results_CHMM(0));
                        Vkappa_tmp = real(results_CHMM(1));
                        sum_Ekappa2_diff_tmp = as_scalar(real(results_CHMM(3)));
                        EKd_tmp = exp(Ekappa_tmp + Vkappa_tmp / 2);
                        EKdinv_tmp = exp(- Ekappa_tmp + Vkappa_tmp / 2);
                        Hkappa_tmp = - results_logdet.logdetK(0) / 2;
                        ELBO_tmp = accu(Ekappa_tmp) - accu(data2.col(j) % EKd_tmp) +
                                accu(pL1pEKdinv_true.col(j) % EKdinv_tmp) - Egamma / 2 *
                                sum_Ekappa2_diff_tmp + Hkappa_tmp;
                        if (ELBO_max > ELBO0_vec(j) && ELBO_tmp <= ELBO_max) {
                            eta *= 2;
                            h_kappa.col(j) += eta * natgrad_h_kappa.col(j);
                            zeta_kappa_d.col(j) += eta * natgrad_zeta_kappa_d.col(j);
                            zeta_kappa_od.col(j) += eta * natgrad_zeta_kappa_od.col(j);
                            Hkappa(j) = Hkappa_max;
                            break;
                        } else {
                            if (ELBO_tmp > ELBO_max) {
                                ELBO_max = ELBO_tmp;
                                Hkappa_max = Hkappa_tmp;
                            }
                            eta /= 2;
                        }
                    }
                    else
                        eta /= 2;
                }
            }
            
            
            results_CHMM = FwdBwdCHMM(cx_mat(h_kappa, mat(N, P, fill::zeros)),
                    zeta_kappa_d, zeta_kappa_od, N, P);
            Ekappa = real(results_CHMM(0));
            Vkappa = real(results_CHMM(1));
            COVkappa = real(results_CHMM(2));
            sum_Ekappa2_diff = real(results_CHMM(3));
            EKd_mat = exp(Ekappa + Vkappa / 2);
            EKdinv_mat = exp(- Ekappa + Vkappa / 2);
        }
        
        // update s_init and A
        a = 1.0 + Pe - accu(Es_mat.row(0));
        b = 1.0 + accu(Es_mat.row(0));
        log_s_init(0) = boost::math::digamma(a) - boost::math::digamma(a + b);
        log_s_init(1) = boost::math::digamma(b) - boost::math::digamma(a + b);
        
        c0 = 1.0 + accu(sum_pair_density.col(0));
        d0 = 1.0 + accu(sum_pair_density.col(1));
        c1 = 1.0 + accu(sum_pair_density.col(3));
        d1 = 1.0 + accu(sum_pair_density.col(2));
        
        log_A(0, 0) = boost::math::digamma(c0) - boost::math::digamma(c0 + d0);
        log_A(0, 1) = boost::math::digamma(d0) - boost::math::digamma(c0 + d0);
        log_A(1, 1) = boost::math::digamma(c1) - boost::math::digamma(c1 + d1);
        log_A(1, 0) = boost::math::digamma(d1) - boost::math::digamma(c1 + d1);
        
        // update Egamma
        Egamma = (N - 1.0) * P / accu(sum_Ekappa2_diff);
        
        if (iter <= anneal_iters) {
            pb.update();
            if (iter % 10 == 0) {
                
                mexEvalString("drawnow;");
                bandwidth = bandwidth - bandwidth_diff;
                if (bandwidth < 1.0) bandwidth = 1.0;
                lb = regspace(0, N - 1) - ceil(bandwidth) + 0.5;
                lb(find(lb < -0.5)).fill(-0.5);
                ub = regspace(0, N - 1) + ceil(bandwidth) - 0.5;
                ub(find(ub > N - 0.5)).fill(N - 0.5);
                bd_range = ub - lb;
                T = (bandwidth_ub - 1.0) / (bandwidth_ub - bandwidth);
            }
            
            if (iter == anneal_iters) {
                mexPrintf("\n");
                id_rnd.clear();
                data_rnd.clear();
                data_rnd2.clear();
                sum_EKod_m_data_rnd.clear();
                pL1pEKod.clear();
                pL1pVKodmn2.clear();
                pL1pEs_true.clear();
                Hs.clear();
                Es_tmp.clear();
                sum_pair_density_tmp.clear();
                pL1pEKdinv.clear();
                Egamma_rnd.clear();
                
                T = 1.0;
                Es_mat_old = Es_mat;
                EKod_mat_old = EKod_mat;
                EKd_mat_old = EKd_mat;
            }
            
        } else {
            diff_s = max(abs(vectorise(Es_mat - Es_mat_old)));
            diff_d = EKd_mat - EKd_mat_old;
            diff_od = EKod_mat - EKod_mat_old;
            diff_d_max = max(abs(vectorise(diff_d)));
            diff_d_r = sqrt(mean(vectorise(square(diff_d))) / mean(vectorise(square(EKd_mat))));
            diff_od_max = max(abs(vectorise(diff_od)));
            diff_od_r = sqrt(mean(vectorise(real(diff_od % conj(diff_od)))) / 
                    mean(vectorise(real(EKod_mat % conj(EKod_mat)))));
            mexPrintf("iteration %3d: diff_s=%e, diff_d_r=%e, diff_d_max=%e, diff_od_r=%e, diff_od_max=%e\n", iter,
                    diff_s, diff_d_r, diff_d_max, diff_od_r, diff_od_max);
            mexEvalString("drawnow;");
            
            if (diff_od_r < tol_relative && diff_d_r < tol_relative && diff_s < tol_s) break;
            else {
                Es_mat_old = Es_mat;
                EKod_mat_old = EKod_mat;
                EKd_mat_old = EKd_mat;
            }
        }
        
        
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double run_time = std::chrono::duration<double, std::milli>(t_end-t_start).count() / 1e3;
    if (iter < max_iter) mexPrintf("BADGE converges, elapsed time is %f seconds.\n",run_time);
    else mexPrintf("BADGE reaches the maximum number of iterations, elapsed time is %f seconds.\n",run_time);
    
    //Output to MATLAB
    plhs[0] = armaCreateMxMatrix(N, P, mxDOUBLE_CLASS,mxREAL);
    armaSetPr(plhs[0], EKd_mat);
    plhs[1] = armaCreateMxMatrix(N, Pe, mxDOUBLE_CLASS, mxCOMPLEX);
    armaSetCx(plhs[1], EKod_mat);
    plhs[2] = armaCreateMxMatrix(N, Pe, mxDOUBLE_CLASS, mxCOMPLEX);
    armaSetCx(plhs[2], EJ_mat);
    plhs[3] = armaCreateMxMatrix(N, Pe, mxDOUBLE_CLASS,mxREAL);
    armaSetPr(plhs[3], Es_mat);
    plhs[4] = mxCreateDoubleScalar(run_time);
    
//     return List::create(Named("EKd_mat") = EKd_mat, Named("EKod_mat") = EKod_mat, Named("EJ_mat") = EJ_mat, Named("Es_mat") = Es_mat,
//             Named("data") = data, Named("run_time") = run_time);
}
