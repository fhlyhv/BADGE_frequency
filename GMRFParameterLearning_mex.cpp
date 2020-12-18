#include "armaMex.hpp"
#include "GMRFParameterLearning.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    cx_mat K0 = armaGetCx(prhs[0]);
    cx_mat S = armaGetCx(prhs[1]);
    uvec idr = conv_to<uvec>::from(armaGetPr(prhs[2]));
    uvec idc = conv_to<uvec>::from(armaGetPr(prhs[3]));
    uword max_iter;
    if (nrhs > 4) max_iter = (int)armaGetDouble(prhs[4]); 
    else max_iter = 1000; 
    
    cx_mat K;
    uword P = K0.n_cols;
    
    K = GMRFParameterLearning(K0, S, idr, idc, max_iter);
    
    plhs[0] = armaCreateMxMatrix(P, P, mxDOUBLE_CLASS, mxCOMPLEX);
    armaSetCx(plhs[0], K);
}