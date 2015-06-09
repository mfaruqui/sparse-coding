CC = g++
INCLUDES =  -I/usr/include/eigen3/
CFLAGS = -std=c++11 -g -O3 -ffast-math 
LIBS = -fopenmp
CNPY = -Lcnpy cnpy/libcnpy.a
TARGET = sparse nonneg sparse-cnpy nonneg-cnpy
all: $(TARGET)

sparse: sparse.cc utils.cc
	$(CC) $(INCLUDES) $(CFLAGS) $^ -o $@ $(LIBS)
nonneg: sparse-nonneg.cc utils.cc
	$(CC) $(INCLUDES) $(CFLAGS) $^ -o $@ $(LIBS)
sparse-cnpy: sparse-cnpy.cc utils.cc
	$(CC) $(INCLUDES) $(CFLAGS) $^ -o $@ $(LIBS) $(CNPY)
nonneg-cnpy: nonneg-cnpy.cc utils.cc
	$(CC) $(INCLUDES) $(CFLAGS) $^ -o $@ $(LIBS) $(CNPY)
clean:
	$(RM) $(TARGET) *~
