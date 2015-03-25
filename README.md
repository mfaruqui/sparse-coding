#Sparse Coding
Manaal Faruqui, mfaruqui@cs.cmu.edu

This tool implements sparse coding (Lee et al, 2006) for converting dense word vector representations to highly sparse vectors. The implementation can be run on multiple cores in parallel with asynchronous updates. The sparsity is introduced in the word vectors using L1 regularization.

###Data you need

1. Word Vector File

Each vector file should have one word vector per line as follows (space delimited):-

```the -1.0 2.4 -0.3 ...```

###Compile

You need to download the latest Eigen stable release from here: http://eigen.tuxfamily.org/index.php?title=Main_Page

Unzip the folder and provide its path in the makefile:
INCLUDES = -I PATH_TO_EIGEN

After this just execute the following command:

```make``` for sparse coding
```make nonneg``` for non-negative sparse coding
