include_paths = {'-I./', ['-I', fullfile(pwd, './armadillo/include')], ...
    ['-I', fullfile(pwd, './armadillo/mex_interface')], ...
    '-I/opt/intel/mkl/include'};

library_paths = {'-L/opt/intel/compilers_and_libraries/linux/lib/intel64'};

CXXFLAGS = 'CXXFLAGS="$CXXFLAGS -m64 -fopenmp -DMKL_ILP64 -DARMA_DONT_USE_WRAPPER"';
CXXOPTIMFLAGS = 'CXXOPTIMFLAGS="-O3 -fwrapv -DNDEBUG"';

LDFLAGS = 'LDFLAGS="$LDFLAGS -fopenmp"';
LDOPTIMFLAGS = 'LDOPTIMFLAGS="-O3"';

libraries = {'-liomp5', '-lpthread', '-ldl'};
CXXLIBS = 'CXXLIBS="$CXXLIBS -Wl,--start-group /opt/intel/mkl/lib/intel64/libmkl_intel_ilp64.a /opt/intel/mkl/lib/intel64/libmkl_intel_thread.a /opt/intel/mkl/lib/intel64/libmkl_core.a -Wl,--end-group"';

source_files = {'FwdBwdCHMM_var.cpp', 'FwdBwdCHMM.cpp', 'FwdBwdDHMM.cpp', 'LogDetTriDiag.cpp', 'SumEKodMDataUpdate.cpp', 'CrossValidation.cpp', ...
    'GMRFParameterLearning.cpp', 'CrossValidation_QUIC.cpp', 'BADGE_cpp.cpp'};
mex(include_paths{:}, CXXFLAGS, CXXOPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, library_paths{:}, libraries{:}, CXXLIBS, source_files{:}, '-output', 'BADGE.mexa64'); 