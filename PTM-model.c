#include "PTM-model.h"
/* PTM-model.c
 *
 * Parsimonious Topic Models with Salient Word Discovery
 *
 * Author: Hossein Soleimani and David J. Miller
 *         Department of Electrical Engineering
 *         The Pennsylvania State University, University Park
 *         hsoleimani@psu.edu, djmiller@engr.psu.edu
 *
 * The current version is last modified on 01-20-2014. Please note that
 * this program is still developmental.
 *
 * For details of the algorithm, please check the paper,
 * Hossein Soleimani and David J. Miller,
 * "Parsimonious Topic Models with Salient Word Discovery", arXiv:1401.6169.
 *
 * This program is free program; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or any later
 * version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even he implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 */


ptm_alpha* new_ptm_alpha(int ndocs, int ntopics){
    int d,j;
    ptm_alpha* alpha;
    alpha = malloc(sizeof(ptm_alpha));
    alpha->alpha = malloc(sizeof(double*)*(ndocs));
    alpha->v = malloc(sizeof(int*)*(ndocs));
    for (d = 0; d < ndocs; d++){
    	alpha->alpha[d] = malloc(sizeof(double) * ntopics);
    	alpha->v[d] = malloc(sizeof(int) * ntopics);
    }
    for (d = 0; d< ndocs; d++){
    	for (j = 0; j < ntopics; j++){
    		alpha->alpha[d][j] = 0;
    		alpha->v[d][j] = 0;
    	}
    }

    return(alpha);
}

void random_initialize_model(ptm_model* model, corpus* c)
{
    int num_topics = model->num_topics;
    double sum=0;
    int j, n, d;
    double *wbar_sum, *mu, temp, total;

    wbar_sum = malloc(sizeof(double)*num_topics);
    mu = malloc(sizeof(double)*num_topics);

    //random initialization
    sum = 0.0;
  	for (d = 0; d < c->num_docs; d++){
		for (n = 0; n < c->docs[d].length; n++){
			model->prob0_w[c->docs[d].words[n]] += c->docs[d].counts[n]+1e-10;
			sum += c->docs[d].counts[n]+1e-10;
		}
	}
	for (n = 0; n < model->num_terms; n++){
		model->prob0_w[n] = model->prob0_w[n]/sum;
	}


	for (j = 0; j < num_topics; j++){
		wbar_sum[j] = 0.0;
		mu[j] = 0.0;
		total = 0.0;
		//randomly choose NUM_INIT documents for each topic
		for (n = 0; n < model->num_terms; n++){
			temp = myrand();
			if (temp < 0.1){
				model->u[j][n] = 1;
				temp = myrand();
				model->prob_w[j][n] = temp;
				total += temp;
			}
			else{
				model->prob_w[j][n] = 0.0;
				model->u[j][n] = 0;
			}
			wbar_sum[j] += model->u[j][n]*model->prob_w[j][n];
		}
	}

	for (n = 0; n < model->num_terms; n++){
		for (j = 0; j < num_topics; j++){
			mu[j] += (1-model->u[j][n])*model->prob0_w[n];
		}
	}

	for  (j = 0; j < num_topics; j++){
		mu[j] = wbar_sum[j]/(1-mu[j]);
		for (n = 0; n < model->num_terms; n++){
			model->prob_w[j][n] = model->prob_w[j][n]/mu[j];
		}
	}

	free(wbar_sum);
    free(mu);

}

void random_initialize_alpha(ptm_model* model, ptm_alpha* alpha, corpus* c)
{
    double lkmax, temp, lk, sum;
    int j,n,d,jmax, i, again, cnt, dd;
    document* doc;
    int * sdocs = malloc(sizeof(int)*c->num_docs);
    int * best_topics = malloc(sizeof(int)*model->num_topics);
    int num_best = 0;

    // starting with only 1 topic on
    for (d = 0; d < c->num_docs; d++){

		lkmax = -1e15;
		jmax = 0;
		for (j = 0; j < model->num_topics; j++){
			alpha->alpha[d][j] = 0.0;
			alpha->v[d][j] = 0;
		}
		doc = &(c->docs[d]);
		num_best = 0;
		for (j = 0; j < model->num_topics; j++){
			lk = 0.0;
			for (n = 0; n < c->docs[d].length; n++){
				if ((model->u[j][doc->words[n]] == 0) | (model->prob_w[j][doc->words[n]] == 0))
					temp = model->prob0_w[doc->words[n]];
				else
					temp = model->prob_w[j][doc->words[n]];
				if (temp != 0){
					lk += doc->counts[n]*log(temp);
				}
			}
			if (lk == lkmax){
				best_topics[num_best] = j;
				num_best += 1;
			}
			if (lk > lkmax){
				jmax = j;
				lkmax = lk;
				num_best = 1;
				best_topics[0] = j;
			}
		}
		if (num_best > 1){
			jmax = best_topics[(int) floor((myrand())*num_best)];
		}
		alpha->alpha[d][jmax] = 1.0;
		alpha->v[d][jmax] = 1;
	}
    free(sdocs);
    free(best_topics);
}

void random_initialize_alpha_test(ptm_model* model, ptm_alpha* alpha, corpus* c)
{
    double sum=0.0;
    int j, d;

   for (d = 0; d < c->num_docs; d++){
    	sum = 0.0;
    	for (j = 0; j < model->num_topics; j++){
    		alpha->alpha[d][j] = myrand();
    		sum += alpha->alpha[d][j];
    	}
    	for (j = 0; j < model->num_topics; j++){
			alpha->alpha[d][j] = alpha->alpha[d][j]/sum;
			alpha->v[d][j] = 1;
		}
    }


}


void corpus_initialize_model(ptm_alpha* alpha, ptm_model* model, corpus* c){

    int num_topics = model->num_topics;
    int i, k, d, n,j,jmax,dd,total;
    double sum,temp, lk,lkmax;
    int cnt, again;
    document* doc;

    int * best_topics = malloc(sizeof(int)*model->num_topics);
    int num_best;

    double *wbar_sum, *mu;
    int *sdocs;

    wbar_sum = malloc(sizeof(double)*num_topics);
    mu = malloc(sizeof(double)*num_topics);
    sdocs = malloc(sizeof(int)*c->num_docs);

    cnt=0;

    // initialize prob0_w
    sum = 0.0;
  	for (d = 0; d < c->num_docs; d++){
  		for (n = 0; n < c->docs[d].length; n++){
  			model->prob0_w[c->docs[d].words[n]] += c->docs[d].counts[n]+1e-10;
  			sum += c->docs[d].counts[n]+1e-10;
  		}
  	}
  	for (n = 0; n < model->num_terms; n++){
  		model->prob0_w[n] = model->prob0_w[n]/sum;
  	}

  	// initialize other topics
    for (k = 0; k < num_topics; k++){
    	wbar_sum[k] = 0.0;
    	mu[k] = 0.0;
    	total = 0.0;
    	//randomly choose NUM_INIT documents for each topic
        for (i = 0; i < NUM_INIT; i++){
            again = 1;
            while(again){
            	again = 0;
				d = floor(myrand() * c->num_docs);
				for (dd = 0; dd < cnt; dd++){
					if (sdocs[dd] == d){
						again = 1;
						break;
					}
				}
				if (again == 0){
					sdocs[cnt] = d;
					cnt++;
				}
            }
            alpha->alpha[d][k] = 1.0;
            alpha->v[d][k] = 1;
            doc = &(c->docs[d]);
            for (n = 0; n < doc->length; n++){
                model->prob_w[k][doc->words[n]] += doc->counts[n]+0.1;
                model->u[k][doc->words[n]] = 1;
                total += doc->counts[n]+0.1;
            }
        }
        for (n = 0; n < model->num_terms; n++){
        	if (model->prob_w[k][n] == 0){
        		model->u[k][n] = 0;
        	}
        	wbar_sum[k] += model->u[k][n]*model->prob_w[k][n];
        }
    }

    for (n = 0; n < model->num_terms; n++){
		for (j = 0; j < num_topics; j++){
			mu[j] += (1-model->u[j][n])*model->prob0_w[n];
		}
	}

	for  (j = 0; j < num_topics; j++){
		mu[j] = wbar_sum[j]/(1-mu[j]);
		for (n = 0; n < model->num_terms; n++){
			model->prob_w[j][n] = model->prob_w[j][n]/mu[j];
		}
	}

	// assign other documents to the topics
    for (d = 0; d < c->num_docs; d++){
		lkmax = -1e15;
		jmax = 0;
		//check if it's not initialized
		sum = 0;
		for (j = 0; j < model->num_topics; j++){
			sum += alpha->alpha[d][j];
		}
		if (sum == 1) continue;
		doc = &(c->docs[d]);
		num_best = 0;
		for (j = 0; j < model->num_topics; j++){
			lk = 0;
			for (n = 0; n < c->docs[d].length; n++){
				if (model->u[j][doc->words[n]] == 1)
					temp = model->prob_w[j][doc->words[n]];
				else
					temp = model->prob0_w[doc->words[n]];
				if (temp != 0){
					lk += doc->counts[n]*log(temp);
				}
			}
			if (lk == lkmax){
				best_topics[num_best] = j;
				num_best += 1;
			}
			if (lk > lkmax){
				jmax = j;
				lkmax = lk;
				num_best = 1;
				best_topics[0] = j;
			}
		}
		if (num_best > 1){
			jmax = best_topics[(int)floor(myrand()*num_best)];
		}
		alpha->alpha[d][jmax] = 1.0;
		alpha->v[d][jmax] = 1;
	}

    //prepare for re-initialization
    for (n = 0; n < model->num_terms; n++){
    	model->prob0_w[n] = 0.0;
    	for (j = 0; j < num_topics; j++){
    		model->prob_w[j][n] = 0.0;
    		model->u[j][n] = 1;
    	}
    }

    for (j = 0; j < num_topics; j++){
    	wbar_sum[j] = 0.0;
    	mu[j] = 0.0;
    	for (d = 0; d < c->num_docs; d++){
    		if (alpha->v[d][j] == 0) continue;
    		for (n = 0; n < c->docs[d].length; n++){
    			model->prob_w[j][c->docs[d].words[n]] += c->docs[d].counts[n] + 0.000001;
    		}
    	}
    	for (n = 0; n < model->num_terms; n++){
    		if (model->prob_w[j][n] == 0)
    			model->u[j][n] = 0;
    		wbar_sum[j] += model->u[j][n]*model->prob_w[j][n];
    	}
    }

    sum = 0.0;
	for (d = 0; d < c->num_docs; d++){
		for (n = 0; n < c->docs[d].length; n++){
			model->prob0_w[c->docs[d].words[n]] +=c->docs[d].counts[n]+1e-10;
			sum +=c->docs[d].counts[n]+1e-10;
		}
	}
    for (n = 0; n < model->num_terms; n++){
    	model->prob0_w[n] = model->prob0_w[n]/sum;
    	for (j = 0; j < num_topics; j++){
    		mu[j] += (1-model->u[j][n])*model->prob0_w[n];
    	}
    }

    for  (j = 0; j< num_topics; j++){
    	mu[j] = wbar_sum[j]/(1-mu[j]);
    	for (n = 0; n < model->num_terms; n++){
    		model->prob_w[j][n] = model->prob_w[j][n]/mu[j];
    	}
    }

    free(wbar_sum);
    free(mu);
    free(sdocs);
    free(best_topics);
}

ptm_model* new_ptm_model(int num_terms, int num_topics)
{
    int j,n;
    ptm_model* model;

    model = malloc(sizeof(ptm_model));
    model->num_topics = num_topics;
    model->num_terms = num_terms;
    model->u=malloc(sizeof(int*)*(num_topics));
    model->prob_w = malloc(sizeof(double*)*num_topics);
    model->prob0_w = malloc(sizeof(double)*num_terms);
    for (j = 0; j < num_topics; j++){
    	model->prob_w[j] = malloc(sizeof(double)*num_terms);
    	model->u[j] = malloc(sizeof(int)*num_terms);
    	for (n = 0; n < num_terms; n++){
    		model->prob0_w[n] = 0.0;
    		model->prob_w[j][n] = 0.0;
    		model->u[j][n] = 1;
    	}
    }
    return(model);
}


ptm_emvars* new_ptm_emvars(int nterms, int ntopics, int ndocs){

	int j,n;

	ptm_emvars * emvars = malloc(sizeof(ptm_emvars));
	emvars->m_d = malloc(sizeof(double)*ndocs);

	emvars->wbar = malloc(sizeof(double*)*ntopics);
	emvars->t1_sum = malloc(sizeof(double)*ntopics);
	emvars->sum_vjd2 = malloc(sizeof(double)*ntopics);
	emvars->alpha_temp = malloc(sizeof(double)*ntopics);
	emvars->alpha_prev = malloc(sizeof(double)*ntopics);
	emvars->gamma = malloc(sizeof(double)*ntopics);
	emvars->gamma1 = malloc(sizeof(double)*ntopics);
	emvars->gamma2 = malloc(sizeof(double)*ntopics);
	emvars->prob_w_sum = malloc(sizeof(double)*ntopics);
	emvars->wbar_sum = malloc(sizeof(double)*ntopics);
	emvars->mu = malloc(sizeof(double)*ntopics);
	emvars->ldvjd_sum = malloc(sizeof(double)*ntopics);
	emvars->ldvjd_sum_prev = malloc(sizeof(double)*ntopics);
	emvars->prob0_w_temp = malloc(sizeof(double)*nterms);
	emvars->tpcpermute = malloc(sizeof(int)*ntopics);
	emvars->wrdpermute = malloc(sizeof(int)*nterms);
	for (j = 0; j < ntopics; j++){

		emvars->t1_sum[j] = 0.0;
		emvars->sum_vjd2[j] = 0.0;
		emvars->alpha_temp[j] = 0.0;
		emvars->alpha_prev[j] = 0.0;
		emvars->gamma[j] = 0.0;
		emvars->gamma1[j] = 0.0;
		emvars->gamma2[j] = 0.0;
		emvars->prob_w_sum[j] = 0.0;
		emvars->wbar_sum[j] = 0.0;
		emvars->mu[j] = 0.0;
		emvars->ldvjd_sum[j] = 0.0;
		emvars->ldvjd_sum_prev[j] = 0.0;
		emvars->wbar[j] = malloc(sizeof(double)*nterms);
		emvars->tpcpermute[j] = 0;
		for (n = 0; n < nterms; n++){
			emvars->wbar[j][n] = 0.0;
			emvars->prob0_w_temp[n] = 0.0;
			emvars->wrdpermute[n] = 0;
		}
	}
	return(emvars);
}


void free_ptm_emvars(ptm_emvars* emvars, int ntopics){

	int j;

	free(emvars->t1_sum);
	free(emvars->sum_vjd2);
	free(emvars->alpha_temp);
	free(emvars->alpha_prev);
	free(emvars->gamma);
	free(emvars->gamma1);
	free(emvars->gamma2);
	free(emvars->prob_w_sum);
	free(emvars->wbar_sum);
	free(emvars->mu);
	free(emvars->ldvjd_sum);
	free(emvars->ldvjd_sum_prev);
	free(emvars->prob0_w_temp);
	free(emvars->m_d);
	for (j = 0; j < ntopics; j++){
		free(emvars->wbar[j]);
	}
}


void save_ptm_model(ptm_model* model, char* model_root)
{
    char filename[100];
    FILE* fileptr;
    int j,n;

    sprintf(filename, "%s.beta", model_root);
    fileptr = fopen(filename, "w+");
    for (n = 0; n < model->num_terms; n++){
    	fprintf(fileptr, "%5.10f", log(model->prob0_w[n]));
    	for (j = 0; j < model->num_topics; j++){
		if (model->prob_w[j][n]==0)
		    fprintf(fileptr, " %5.10f", model->prob_w[j][n]);
		else
	            fprintf(fileptr, " %5.10f", log(model->prob_w[j][n]));
	}
    	fprintf(fileptr, "\n");
    }
    fflush(fileptr);
    fclose(fileptr);
    sprintf(filename, "%s.u", model_root);
	fileptr = fopen(filename, "w+");
	for (n = 0; n < model->num_terms; n++){
		for (j = 0; j < model->num_topics; j++)
			fprintf(fileptr, " %d", model->u[j][n]);
		fprintf(fileptr, "\n");
	}
	fflush(fileptr);
	fclose(fileptr);
    sprintf(filename, "%s.other", model_root);
    fileptr = fopen(filename, "w");
    fprintf(fileptr, "num_topics %d\n", model->num_topics);
    fprintf(fileptr, "num_terms %d\n", model->num_terms);
    fclose(fileptr);
}



ptm_model* load_ptm_model(char* model_root)
{
    char filename[100];
    FILE* fileptr;
    int j, num_terms, num_topics, n;
    float x,y;
    int u;

    sprintf(filename, "%s.other", model_root);
    printf("loading %s\n", filename);
    fileptr = fopen(filename, "r");
    fscanf(fileptr, "num_topics %d\n", &num_topics);
    fscanf(fileptr, "num_terms %d\n", &num_terms);
    fclose(fileptr);

    ptm_model* model = new_ptm_model(num_terms, num_topics);
    model->num_terms=num_terms;
    model->num_topics=num_topics;

    sprintf(filename, "%s.beta", model_root);
    printf("loading %s\n", filename);
    fileptr = fopen(filename, "r");
    for (n = 0; n < num_terms; n++){
    	fscanf(fileptr, "%f", &y);
		model->prob0_w[n] = exp(y);
        for (j = 0; j < num_topics; j++){
            fscanf(fileptr, " %f", &x);
	    if (x==0)
	    	model->prob_w[j][n] = x;
	    else
	        model->prob_w[j][n] = exp(x);
        }
    }
    fclose(fileptr);
    sprintf(filename, "%s.u", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	for (n = 0; n < num_terms; n++){
		for (j = 0; j < num_topics; j++){
			fscanf(fileptr, "%d ", &u);
			model->u[j][n]=u;
		}
	}
	fclose(fileptr);
    return(model);
}


void save_ptm_alpha(char* root, ptm_alpha* alpha, int num_docs, int num_topics)
{
    FILE* fileptr;
    int d, k;
    char filename[100];

    sprintf(filename, "%s.alpha", root);
    fileptr = fopen(filename, "w+");
    for (d = 0; d < num_docs; d++){
    	fprintf(fileptr, "%5.10f", alpha->alpha[d][0]);
    	for (k = 1; k < num_topics; k++){
    		fprintf(fileptr, " %5.10f", alpha->alpha[d][k]);
    	}
    	fprintf(fileptr, "\n");
    }
    fflush(fileptr);
    fclose(fileptr);
    sprintf(filename, "%s.v", root);
    fileptr = fopen(filename, "w+");
	for (d = 0; d < num_docs; d++){
		fprintf(fileptr, "%d", alpha->v[d][0]);
		for (k = 1; k < num_topics; k++){
			fprintf(fileptr, " %d", alpha->v[d][k]);
		}
		fprintf(fileptr, "\n");
	}
	fflush(fileptr);
	fclose(fileptr);
}
