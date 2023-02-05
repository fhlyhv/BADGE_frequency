# BASS (frequency domain)
This matlab toolbox implements the BASS algorithm (formally named BADGE) for learning graphical models from multivariate stationary time series in frequency domain. Please refer to [1] for more details of the algorithm.

The R package for learning time-varying graphical models in the time domain can be found at https://github.com/fhlyhv/BADGE.

## Compiling the mex code

The C++ code is written based on the template-based C++ library Armadillo [2]. 
To achieve the best performance, it is better to link the C++ code with
BLAS and LAPACK in Intel MKL in Linux, since currently openmp 3.1 is not 
supported in Windows.

The mex code for Matlab 2017b and above in both windows and ubuntu OS have 
been provided. 

In the case the mex code are obselete, you can compile the original C++ 
code into the mex code following the instructions below.

Before mex the C++ code, please download and install Intek MKL from
https://software.intel.com/en-us/mkl

In particualr for ubuntu OS, Intel MLK can be downloaded and installed by 
running the following commands in the terminal:

```
cd /tmp  
sudo wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB  
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB  
sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'  
sudo apt-get update  
sudo apt-get install intel-mkl-64bit-2018.2-046
```

To mex the C++ code in Windows using the msvc compiler, please run 
win_msvc_IntelMKL_mex.m in Matlab.

To mex the C++ code in Linux using the g++ compiler, please run 
linux_gpp_IntelMKL_mex.m in Matlab.

You may need to change directory of Intel MKL to your own installation 
directory in the above m files.

## Example

Please refer to main.m for an example of how to call BADGE to analyze a synthetic data set. To test the algorithm on your own data, please replace the matrix `XDat` in main.m by your data. Note that 'XDat' is a N x P matrix, where N is the length of the time series and P is the number of variables.

For more details of how to manipulate the BADGE function. Please refer to the comments in BADGE_cpp.cpp.


[1] H. Yu, S. Wu and J. Dauwels, Efficient Variational Bayes Learning of Graphical Models With Smooth Structural Changes, IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, pp. 475 - 488, 2023.

[2] C. Sanderson, R. Curtin. Armadillo: a template-based C++ library for linear algebra. Journal of Open Source Software, 2016.
