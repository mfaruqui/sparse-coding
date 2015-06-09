#Sparse Coding
Manaal Faruqui, mfaruqui@cs.cmu.edu

This tool implements sparse coding for converting dense word vector representations to highly sparse vectors. The implementation can be run on multiple cores in parallel with asynchronous updates. The sparsity is introduced in the word vectors using L1 regularization. For technical details please refer to Faruqui et al (2015).

###Data you need

Word Vector File. Each vector file should have one word vector per line as follows (space delimited):-

```the -1.0 2.4 -0.3 ...```

###Compile

You need to download the latest Eigen stable release from here: http://eigen.tuxfamily.org/index.php?title=Main_Page

Unzip the folder and provide its path in the makefile:
INCLUDES = -I PATH_TO_EIGEN

After this just execute the following command:

For sparse coding: ```make```

For non-negative sparse coding: ```make nonneg```

###Running the executable

For sparse coding: ```sparse.o```

For non-negative sparse coding: ```nonneg.o```

Usage: ```./sparse.o vec_corpus factor l1_reg l2_reg num_cores outfilename```

Example: ```./sparse.o sample_vecs.txt 10 0.5 1e-5 1 out_vecs.txt```

This example would expand the vectors in sample_vecs.txt to 10 times their original length.

###Reference

```
@InProceedings{faruqui:2015:sparse,
  author    = {Faruqui, Manaal and Tsvetkov, Yulia and Yogatama, Dani and Dyer, Chris and Smith, Noah A.},
  title     = {Sparse Overcomplete Word Vector Representations},
  booktitle = {Proceedings of ACL},
  year      = {2015},
}
```
