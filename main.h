#ifndef MAIN_H
#define MAIN_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "PTM.h"
#include "PTM-model.h"
#include "cokus.h"
#include <getopt.h>

double BIAS = 1e-15;
double pi = 3.1416;
int NTOPICS;
int SAVE_LAG;
float EM_CONVERGED;
int EM_MAX_ITER;
int STEP;


void write_wrd_asgnmnts(char* filename, corpus* c,
		ptm_model* model, ptm_alpha* alpha, ptm_emvars*);
void em(corpus* c, ptm_model* model, ptm_alpha* alpha, int switchupdate, ptm_emvars*);
void em_infer(corpus* c, ptm_model* model, ptm_alpha* alpha, double * alpha_temp, double* gamma);
void train(char* start, char* directory, corpus* corpus, char* model_dir);
double compute_likelihood(ptm_alpha* alpha, ptm_model* model,corpus* corpus,double* doc_likelihood);
void inference(char* model_root, char* directory, corpus* corpus);
corpus* read_data(char* data_filename);

void random_permute(int size, int* vec);

#endif


