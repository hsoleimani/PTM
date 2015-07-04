
#ifndef PTM_H
#define PTM_H

typedef struct
{
    int* words;
    int* counts;
    int length;
    int total;
} document;


typedef struct
{
    document* docs;
    int num_terms;
    int num_docs;
} corpus;

typedef struct
{
    int num_topics;
    int num_terms;
    double** prob_w;
    double* prob0_w;
    short** u;
} ptm_model;


typedef struct
{
    double** alpha;
    short** v;
} ptm_alpha;

typedef struct
{
	double *t1_sum;
	double *sum_vjd2;
	double *alpha_temp;
	double *alpha_prev;
	double *prob0_w_temp;
	double *gamma;
	double *gamma1;
	double *gamma2;
	double *prob_w_sum;
	double *wbar_sum;
	double *mu;
	double *ldvjd_sum;
	double *ldvjd_sum_prev;
	double **wbar;
	double *m_d;
	int *wrdpermute;
	int *tpcpermute;
}ptm_emvars;

#endif
