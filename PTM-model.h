#ifndef PTM_MODEL_H
#define PTM_MODEL_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "PTM.h"
#include "cokus.h"

#define myrand() (double) (((unsigned long) randomMT()) / 4294967296.)
#define NUM_INIT 5

void free_plda_model(ptm_model*);
void save_ptm_model(ptm_model*, char*);
ptm_model* new_ptm_model(int, int);
ptm_emvars* new_ptm_emvars(int , int ,int);

ptm_alpha* new_ptm_alpha(int ndocs, int ntopics);
ptm_alpha* load_ptm_alpha(char* model_root, int ntopics, int ndocs,ptm_alpha* alpha);
void save_ptm_alpha(char* filename, ptm_alpha* alpha, int num_docs,int num_topics);
void corpus_initialize_model(ptm_alpha* alpha, ptm_model* model, corpus* c);
void random_initialize_model(ptm_model* model, corpus* c);
void random_initialize_alpha(ptm_model* model, ptm_alpha* alpha, corpus* c);
void random_initialize_alpha_test(ptm_model* model, ptm_alpha* alpha, corpus* c);
ptm_model* load_ptm_model(char* model_root);
void free_ptm_emvars(ptm_emvars* emvars, int ntopics);

#endif
