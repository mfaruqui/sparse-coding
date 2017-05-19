#Sparse Coding
Manaal Faruqui, mfaruqui@cs.cmu.edu
.npy file support njh@njhurst.com

This tool implements sparse coding for converting dense word vector representations to highly sparse vectors. The implementation can be run on multiple cores in parallel with asynchronous updates. The sparsity is introduced in the word vectors using L1 regularization. For technical details please refer to Faruqui et al (2015).

###Data you need

Word Vector File. Each vector file should have one word vector per line as follows (space delimited):-

```the -1.0 2.4 -0.3 ...```

###Compile

You need to download the latest Eigen stable release from here: http://eigen.tuxfamily.org/index.php?title=Main_Page

Unzip the folder and provide its path in the makefile:
INCLUDES = -I PATH_TO_EIGEN

also get https://github.com/rogersce/cnpy and put it in the same dir (this might be a good use for git submodules)

After this just execute the following command:

```make all```

For sparse coding: ```make sparse```

For non-negative sparse coding: ```make nonneg```

###Running the executable

For sparse coding: ```sparse```

For non-negative sparse coding: ```nonneg```

Usage: ```./sparse vec_corpus factor l1_reg l2_reg num_cores outfilename```

Example: ```./sparse sample_vecs.txt 10 0.5 1e-5 1 out_vecs.txt```

This example would expand the vectors in sample_vecs.txt to 10 times their original length.

###npy files
put the file in a directory:
```sample/vector.npy```

(I put the vocab in the same dir as a txt file, one line per vector: ```sample/vocab.txt```)

Usage: ```./sparse-cnpy vec_corpus_dir factor l1_reg l2_reg num_cores outfilename```

Example: ```./sparse-cnpy sample/ 10 0.5 1e-5 1 out_vecs.txt```

###Reference

```
@InProceedings{faruqui:2015:sparse,
  author    = {Faruqui, Manaal and Tsvetkov, Yulia and Yogatama, Dani and Dyer, Chris and Smith, Noah A.},
  title     = {Sparse Binary Word Vector Representations},
  booktitle = {Proceedings of ACL},
  year      = {2015},
}
```
