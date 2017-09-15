#include "utils.h"

using namespace std;
using namespace Eigen;

/* Try splitting over all whitespaces not just space */
vector<string> split_line(const string& line, char delim) {
  vector<string> words;
  stringstream ss(line);
  string item;
  while (std::getline(ss, item, delim)) {
    if (!item.empty())
      words.push_back(item);
  }
  return words;
}

void ReadVecsFromFile(const string& vec_file_name, mapStrUnsigned* t_vocab,
                      vector<Col>* word_vecs) {
  ifstream vec_file(vec_file_name.c_str());
  mapStrUnsigned& vocab = *t_vocab;
  unsigned vocab_size = 0;
  if (vec_file.is_open()) {
    string line;
    vocab.clear();
    while (getline(vec_file, line)) {
      vector<string> vector_stuff = split_line(line, ' ');
      string word = vector_stuff[0];
      Col word_vec = Col::Zero(vector_stuff.size()-1);
      for (unsigned i = 0; i < word_vec.size(); ++i)
        word_vec(i, 0) = stof(vector_stuff[i+1]);
      vocab[word] = vocab_size++;
      word_vecs->push_back(word_vec);
    }
    cerr << "Read: " << vec_file_name << endl;
    cerr << "Vocab length: " << word_vecs->size() << endl;
    cerr << "Vector length: " << (*word_vecs)[0].size() << endl << endl;
    vec_file.close();

    assert (word_vecs->size() == vocab.size());
  } else {
    cerr << "Could not open " << vec_file;
    exit(0);
  }
}

void ReadVecsFromFile(const string& vec_file_name, mapUnsignedStr* t_vocab,
                      vector<Col>* word_vecs) {
  ifstream vec_file(vec_file_name.c_str());
  mapUnsignedStr& vocab = *t_vocab;
  unsigned vocab_size = 0;
  if (vec_file.is_open()) {
    string line;
    vocab.clear();
    while (getline(vec_file, line)) {
      vector<string> vector_stuff = split_line(line, ' ');
      string word = vector_stuff[0];
      Col word_vec = Col::Zero(vector_stuff.size()-1);
      for (unsigned i = 0; i < word_vec.size(); ++i)
        word_vec(i, 0) = stof(vector_stuff[i+1]);
      vocab[vocab_size++] = word;
      word_vecs->push_back(word_vec);
    }
    cerr << "Read: " << vec_file_name << endl;
    cerr << "Vocab length: " << word_vecs->size() << endl;
    cerr << "Vector length: " << (*word_vecs)[0].size() << endl << endl;
    vec_file.close();

    assert (word_vecs->size() == vocab.size());
  } else {
    cerr << "Could not open " << vec_file_name;
    exit(0);
  }
}

void ElemwiseTanh(Col* v) {
  for (unsigned i = 0; i < v->rows(); ++i)
    (*v)(i, 0) = tanh((*v)(i, 0));
}

/* v is the vector after taking tanh() */
void ElemwiseTanhGrad(const Col &v, Col* g) {
  for (int i = 0; i < v.rows(); ++i)
    (*g)(i, 0) = 1 - pow(v(i, 0), 2);
}

void ElemwiseAndrewNsnl(Col *v) {
  for (int i = 0; i < v->rows(); ++i) {
    double x = (*v)(i, 0);
    if (x) {
      bool flag = (x < 0);
      double y_n = flag ? -x : x;
      for (unsigned i = 0; i < 12; ++i) {
        const double sq = y_n * y_n;
        y_n = (2 * sq * y_n / 3 + x) / (sq + 1);
      }
      (*v)(i, 0) = flag ? -y_n : y_n;
    }
  }
}

void ElemwiseAndrewNsnlGrad(const Col &v, Col* g) {
  for (int i = 0; i < v.rows(); ++i)
    (*g)(i, 0) = 1 / (1 + pow(v(i, 0), 2));
}


double CosineSim(const Col& ci, const Col& cj) {
  return ci.dot(cj)/sqrt(ci.squaredNorm() * cj.squaredNorm());
}
