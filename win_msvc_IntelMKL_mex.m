include_paths = {'-I.', ['-I', fullfile(pwd, './armadillo/include')], ...
    ['-I', fullfile(pwd, './armadillo/mex_interface')], ...
    '-I"C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/include"'};

library_paths = {'-L"C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64_win"', ...
    '-L"C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/compiler/lib/intel64_win"'};

COMPFLAGS = 'COMPFLAGS="$COMPFLAGS /DMKL_ILP64 /openmp /O2 /GL"';
LDFLAGS = 'LDFLAGS="$LDFLAGS /LTCG:INCREMENTAL"';

libraries = {'-lmkl_intel_lp64', '-lmkl_intel_thread', '-lmkl_core', '-llibiomp5md'};

source_files = {'FwdBwdCHMM_var.cpp', 'FwdBwdCHMM.cpp', 'FwdBwdDHMM.cpp', 'LogDetTriDiag.cpp', 'SumEKodMDataUpdate.cpp', 'CrossValidation.cpp', ...
    'GMRFParameterLearning.cpp', 'CrossValidation_QUIC.cpp', 'BADGE_cpp.cpp'};
mex('-R2017b', include_paths{:}, COMPFLAGS, LDFLAGS, library_paths{:}, libraries{:}, source_files{:}, '-output', 'BADGE.mexw64');


% source_files = {'FwdBwdDHMM.cpp','FwdBwdDHMM_mex.cpp'};
% mex('-v', '-R2017b', include_paths{:}, COMPFLAGS, LDFLAGS, library_paths{:}, libraries{:}, source_files{:}, '-output', 'FwdBwdDHMM.mexw64'); 

% 
% source_files = {'LogDetTriDiag.cpp', 'LogDetTriDiag_mex.cpp'};
% mex('-v', '-R2017b', include_paths{:}, COMPFLAGS, LDFLAGS, library_paths{:}, libraries{:}, source_files{:}, '-output', 'LogDetTriDiag.mexw64');
% 
% 
% source_files = {'FwdBwdCHMM.cpp', 'FwdBwdCHMM_mex.cpp'};
% mex('-v', '-R2017b', include_paths{:}, COMPFLAGS, LDFLAGS, library_paths{:}, libraries{:}, source_files{:}, '-output', 'FwdBwdCHMM.mexw64');
% 
% source_files = {'FwdBwdCHMM_mean.cpp', 'FwdBwdCHMM_mean_mex.cpp'};
% mex('-v', '-R2017b', include_paths{:}, COMPFLAGS, LDFLAGS, library_paths{:}, libraries{:}, source_files{:}, '-output', 'FwdBwdCHMM_mean.mexw64');

% source_files = {'FwdBwdCHMM_var.cpp', 'FwdBwdCHMM_var_mex.cpp'};
% mex('-v', '-R2017b', include_paths{:}, COMPFLAGS, LDFLAGS, library_paths{:}, libraries{:}, source_files{:}, '-output', 'FwdBwdCHMM_var.mexw64');


% mex('-v', '-R2017b', include_paths{:}, COMPFLAGS, LDFLAGS, library_paths{:}, libraries{:}, 'CrossValidation_quadkernel.cpp');

% mex('-R2017b', include_paths{:}, COMPFLAGS, LDFLAGS, library_paths{:}, libraries{:}, 'SumEKodMDataUpdate.cpp');

% source_files = {'FwdBwdCHMM.cpp', 'FwdBwdCHMM_var.cpp', 'FwdBwdDHMM.cpp', 'LogDetTriDiag.cpp', 'OffDiagUpdate_T.cpp'};
% mex('-R2017b', include_paths{:}, COMPFLAGS, LDFLAGS, library_paths{:}, libraries{:}, source_files{:}, '-output', 'OffDiagUpdate_T.mexw64');

% source_files = {'FwdBwdCHMM.cpp', 'FwdBwdCHMM_var.cpp', 'FwdBwdDHMM.cpp', 'LogDetTriDiag.cpp', 'OffDiagUpdate.cpp'};
% mex('-R2017b', include_paths{:}, COMPFLAGS, LDFLAGS, library_paths{:}, libraries{:}, source_files{:}, '-output', 'OffDiagUpdate.mexw64');

% source_files = {'FwdBwdCHMM.cpp', 'LogDetTriDiag.cpp', 'DiagUpdate_T.cpp'};
% mex('-R2017b', include_paths{:}, COMPFLAGS, LDFLAGS, library_paths{:}, libraries{:}, source_files{:}, '-output', 'DiagUpdate_T.mexw64');

% source_files = {'FwdBwdCHMM.cpp', 'LogDetTriDiag.cpp', 'DiagUpdate.cpp'};
% mex('-R2017b', include_paths{:}, COMPFLAGS, LDFLAGS, library_paths{:}, libraries{:}, source_files{:}, '-output', 'DiagUpdate.mexw64');