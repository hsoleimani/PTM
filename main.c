#include "main.h"
/* main.c
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

int main(int argc, char* argv[])
{
    corpus* corpus;
    long t1;
    (void) time(&t1);
    seedMT(t1);
    int cc;
    char *task = "training";
    char *dir = NULL;
    char *corpus_file = NULL;
    char *model_name = NULL;
    char *init = "seeded";
    SAVE_LAG = -1;
    EM_CONVERGED = 5e-3;
    EM_MAX_ITER = 100;
    STEP = 0;

    if (argc < 2){
    	printf("***********************Parsimonious Topic Model********************\n");
		printf("usage:\n");
		printf("ptm [options]\n");
		printf("--task:        training/test, default: training\n");
		printf("--directory:   save directory\n");
		printf("--corpus:      corpus file, in lda-c format\n");
		printf("--init:        initialization method; seeded, random, load; default: seeded\n");
		printf("--model:       model name to load (for test or load option)\n");
		printf("--num_topics:  number of topics\n");
		printf("--max_iter:    EM maximum iteration, default: 100\n");
		printf("--convergence: convergence criterion for EM, default: 5e-3\n");
		printf("--save_lag:    save model in intermediate steps, default: -1 (only saves final model)\n");
		printf("--step:        number of topics to remove for next step's initialization (only for model order selection), default: 0 \n");
		printf("\n");
		printf("*******************************************************************\n");
		return(0);
    }

    const char* const short_options = "t:d:c:i:n:m:a:g:s:p";
	const struct option long_options[] = {
		{"task", required_argument, NULL, 't'},
		{"directory", required_argument, NULL, 'd'},
		{"corpus", required_argument, NULL, 'c'},
		{"init", required_argument, NULL, 'i'},
		{"model", required_argument, NULL, 'n'},
		{"num_topics", required_argument, NULL, 'm'},
		{"max_iter", required_argument, NULL, 'a'},
		{"convergence", required_argument, NULL, 'g'},
		{"save_lag", required_argument, NULL, 's'},
		{"step", required_argument, NULL, 'p'},
		{NULL, 0, NULL, 0}
	};
    // read options
	while( (cc = getopt_long(argc, argv, short_options, long_options, NULL)) != -1) {
		switch(cc) {
		case 't':
			task = optarg;
			break;
		case 'd':
			dir = optarg;
			break;
		case 'c':
			corpus_file = optarg;
			break;
		case 'i':
			init = optarg;
			break;
		case 'n':
			model_name = optarg;
			break;
		case 'm':
			NTOPICS = atoi(optarg);
			break;
		case 'a':
			EM_MAX_ITER = atoi(optarg);
			break;
		case 'g':
			EM_CONVERGED = atof(optarg);
			break;
		case 's':
			SAVE_LAG = atoi(optarg);
			break;
		case 'p':
			STEP = atoi(optarg);
			break;
		case '?':
			printf("Unknown parameter\n");
			break;
		default:
			break;
		}
	}

	//write options
	printf("task = %s\n", task);
	printf("directory = %s\n", dir);
	printf("corpus_file = %s\n", corpus_file);
	printf("initialization method = %s\n", init);
	printf("model_name = %s\n", model_name);
	printf("number of topics = %d\n", NTOPICS);
	printf("EM_max_ite = %d\n", EM_MAX_ITER);
	printf("EM_convergence = %f\n", EM_CONVERGED);
	printf("save_lag = %d\n", SAVE_LAG);
	printf("step = %d\n", STEP);

	corpus = read_data(corpus_file);

    // creates directory
	mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

    if (strcmp(task, "training")==0){
		train(init, dir, corpus,model_name);
	}
	else if (strcmp(task, "test")==0){
		inference(model_name, dir, corpus);
    }

    return(0);
}




void train(char* start, char* directory, corpus* corpus, char* model_dir)
{

    int i, j, d, k, nn;
    double cost, likelihood, likelihood_old = 0, converged = 1;
    char filename[200];
    char nbarfilename[200];
    double nbar;
    time_t t1,t2;

    ptm_model *model = NULL;
    ptm_alpha* alpha = NULL;
    ptm_emvars* emvars = NULL;

    emvars = new_ptm_emvars(corpus->num_terms, NTOPICS, corpus->num_docs);

	if (strcmp(start, "seeded")==0){
		model = new_ptm_model(corpus->num_terms, NTOPICS);
		alpha = new_ptm_alpha(corpus->num_docs, NTOPICS);
		corpus_initialize_model(alpha, model, corpus);
	}
	else if (strcmp(start, "load")==0)
	{
		alpha = new_ptm_alpha(corpus->num_docs, NTOPICS);
		model = load_ptm_model(model_dir);
		random_initialize_alpha(model, alpha, corpus);
	}
	else if (strcmp(start, "random")==0)
	{
		printf("Random Initialization...\n");
		model = new_ptm_model(corpus->num_terms, NTOPICS);
		alpha = new_ptm_alpha(corpus->num_docs, NTOPICS);
		random_initialize_model(model, corpus);
		random_initialize_alpha(model, alpha, corpus);
	}


    if (SAVE_LAG > 0){
		sprintf(filename,"%s/000",directory);
		save_ptm_model(model, filename);
		sprintf(filename,"%s/000",directory);
		save_ptm_alpha(filename, alpha, corpus->num_docs, model->num_topics);
    }


    sprintf(nbarfilename, "%s/nbar.txt", directory);
	FILE* nbar_file = fopen(nbarfilename, "w");
    sprintf(filename, "%s/likelihood.dat", directory);
    FILE* likelihood_file = fopen(filename, "w");

    // run expectation maximization

    cost = compute_likelihood(alpha, model, corpus, &likelihood);
    likelihood_old = cost - likelihood;
    time(&t1);
	i = 0;
    while (((converged < 0) || (converged > EM_CONVERGED) || (i <= 2)) && (i <= EM_MAX_ITER))
    {
        i++;
        printf("**** em iteration %d ****\n", i);

        em(corpus,model,alpha,1,emvars);
        cost = compute_likelihood(alpha, model, corpus, &likelihood);
        likelihood = cost - likelihood;
        converged = (likelihood_old - likelihood) / (likelihood_old);
        likelihood_old = likelihood;

        // save model and likelihood
        time(&t2);

        fprintf(likelihood_file, "%10.10f\t%10.10f\t%5.5e\t%5ld\n", likelihood, cost-likelihood, converged, (int) t2-t1);
        nbar = 0.0;
        for (k = 0; k < model->num_topics;k++){
        	for (nn = 0; nn <model->num_terms; nn++){
        		nbar += model->u[k][nn];
        	}
        }
        fprintf(nbar_file, "%f\n", nbar);
        fflush(nbar_file);
        fflush(likelihood_file);
        if ((SAVE_LAG > 0) & ((i % SAVE_LAG) == 0)){
            sprintf(filename,"%s/%03d",directory, i);
            save_ptm_model(model, filename);
            sprintf(filename,"%s/%03d",directory, i);
            save_ptm_alpha(filename, alpha, corpus->num_docs, model->num_topics);
        }
    }

    // update word probabilities and topic proportions after last update of switches
	em(corpus,model,alpha,0,emvars);
	cost = compute_likelihood(alpha, model, corpus, &likelihood);
	likelihood = cost - likelihood;
	converged = (likelihood_old - likelihood) / (likelihood_old);


	// save likelihood
    time(&t2);
	fprintf(likelihood_file, "%10.10f\t%10.10f\t%5.5e\t%5ld\n", likelihood, cost-likelihood, converged, (int) t2-t1);
	fflush(likelihood_file);

    // save the final model
	sprintf(filename,"%s/final",directory);
	save_ptm_model(model, filename);
	sprintf(filename,"%s/final",directory);
	save_ptm_alpha(filename, alpha, corpus->num_docs, model->num_topics);
    // write word assignments
	sprintf(filename,"%s/word-assignments.dat",directory);
	write_wrd_asgnmnts(filename,corpus,model,alpha,emvars);
	fclose(likelihood_file);
    fclose(nbar_file);

    //***********************************************************
    //remove STEP least massive topics and save the model for next step's initialization
    if (STEP > 0) {
    	int *jind;
    	int jj, n, ii, alreadyin;
		double jmin = 1e15;
		double alpha_sum;
		ptm_model *model_temp = NULL;
		jind = malloc(sizeof(int)*STEP);
		jmin = 1e15;
		for (j = 0; j < STEP; j++)
			jind[j] = 0;
		for (i = 0; i < STEP; i++){
			jmin = 1e15;
			for (j = 0; j < NTOPICS; j++){
				alreadyin = 0;
				for (ii = 0;ii < i; ii++){
					if (j == jind[ii]){
						alreadyin = 1;
						break;
					}
				}
				if (alreadyin == 1) continue;
				alpha_sum = 0.0;
				for (d = 0; d < corpus->num_docs; d++){
					alpha_sum += alpha->alpha[d][j];
				}
				if (alpha_sum < jmin){
					jind[i] = j;
					jmin = alpha_sum;
				}
			}
		}

		sprintf(filename,"%s/next",directory);
		model_temp = new_ptm_model(corpus->num_terms, NTOPICS-STEP);

		for (n = 0; n < model->num_terms; n++){
			model_temp->prob0_w[n] = model->prob0_w[n];
			jj = 0;
			for (j = 0; j < model->num_topics; j++){
				alreadyin = 0;
				for (ii = 0; ii < STEP; ii++){
					if (j == jind[ii]){
						alreadyin = 1;
						break;
					}
				}
				if (alreadyin == 0){
					model_temp->prob_w[jj][n] = model->prob_w[j][n];
					model_temp->u[jj][n] = model->u[j][n];
					jj++;
				}

			}
		}
		save_ptm_model(model_temp, filename);
		// free model_temp
		free(model_temp->prob0_w);
		for (j = 0; j < NTOPICS-STEP; j++){
			free(model_temp->prob_w[j]);
			free(model_temp->u[j]);
		}
		free(model_temp);
    	free(jind);
    }


    //free model, alpha, emvars
    free_ptm_emvars(emvars,model->num_topics);
    free(model->prob0_w);
	for (j = 0; j < model->num_topics; j++){
		free(model->prob_w[j]);
		free(model->u[j]);
	}
	free(model);
	for (d = 0; d < corpus->num_docs; d++){
		free(alpha->alpha[d]);
		free(alpha->v[d]);
	}
	free(alpha);


}

void em(corpus* c, ptm_model* model, ptm_alpha* alpha, int switchupdate, ptm_emvars* emvars)
{
    int n, j, d, nn, jj, dd, change, j0, n0;
    double t1, t2, t2_sum, t3, t4;
    double sum_gamma2, sum_gamma;
   	double temp_prob, prev_mu, prev_wbar_sum;
   	int current, counter;
    double dcost, dl, temp1, temp2, dbic;
    double alpha_sum = 0.0;
    double h, tsum, p0t, sum_nj;
    double tsum_vjd,sum_md;
	short tcnt,sum_vjd;
    // init emvars
    for (j = 0; j< model->num_topics; j++){
    	emvars->alpha_temp[j] = 0.0;
    	emvars->prob_w_sum[j] = 0.0;
    	emvars->wbar_sum[j] = 0.0;
    	emvars->mu[j] = 0.0;
    	for (n = 0; n < model->num_terms; n++){
    		emvars->prob0_w_temp[n] = model->prob0_w[n];
    		emvars->wbar[j][n] = 0.0;
    	}
    	emvars->ldvjd_sum[j] = 0.0;
    	emvars->gamma1[j] = 0.0;
    	emvars->gamma2[j] = 0.0;
    	emvars->alpha_temp[j] = 0;
		for (dd=0; dd<c->num_docs; dd++){
			emvars->ldvjd_sum[j] += c->docs[dd].total*alpha->v[dd][j];
		}
    }
	for (n = 0; n <model->num_terms; n++){
		sum_gamma2 = 0.0;
		for (jj = 0; jj < model->num_topics; jj++){
			emvars->gamma1[jj] += model->u[jj][n];
			sum_gamma2 += emvars->ldvjd_sum[jj]*(1-model->u[jj][n]);
		}
		for (jj = 0; jj < model->num_topics; jj++){
			if (sum_gamma2!=0) emvars->gamma2[jj] += (1-model->u[jj][n])/sum_gamma2;
		}
	}

	//main EM loop starts here

    for (d = 0; d < c->num_docs; d++){

    	alpha_sum = 0.0;
    	for (n = 0; n < c->docs[d].length; n++){
    		//compute Pz (gamma) for this word
    		t1 = 0;
    		for (j = 0; j < model->num_topics; j++){
    			if (alpha->v[d][j] == 0){
    				emvars->gamma[j] = 0.0;
    				continue;
    			}
    			if (model->u[j][c->docs[d].words[n]] == 0){
    				emvars->gamma[j] = alpha->v[d][j]*alpha->alpha[d][j]*model->prob0_w[c->docs[d].words[n]];
    			}else
    				emvars->gamma[j] = alpha->v[d][j]*alpha->alpha[d][j]*model->prob_w[j][c->docs[d].words[n]];
    			t1 += emvars->gamma[j];
    		}
    		//if (t1==0){
    		//	printf("%5.10lf, %5.10lf, %d ,%d, %5.10lf\n",model->prob_w[j][c->docs[d].words[n]],model->prob0_w[c->docs[d].words[n]],model->u[j][c->docs[d].words[n]],alpha->v[d][j],alpha->alpha[d][j]);
    		//}
			for (j = 0; j < model->num_topics; j++){
				if (emvars->gamma[j] == 0)
					continue;
				emvars->gamma[j] = emvars->gamma[j]/t1;
				emvars->alpha_temp[j] += emvars->gamma[j]*c->docs[d].counts[n]*alpha->v[d][j];
				alpha_sum += emvars->gamma[j]*c->docs[d].counts[n]*alpha->v[d][j];
				emvars->wbar[j][c->docs[d].words[n]] += emvars->gamma[j]*c->docs[d].counts[n]*alpha->v[d][j];
				emvars->wbar_sum[j] += emvars->gamma[j]*c->docs[d].counts[n]*alpha->v[d][j]*model->u[j][c->docs[d].words[n]];
			}
    	}

		for (j = 0; j < model->num_topics; j++){
			emvars->alpha_temp[j] = emvars->alpha_temp[j]/alpha_sum;
			alpha->alpha[d][j] = emvars->alpha_temp[j];
			emvars->alpha_temp[j] = 0.0;
		}
    }

    for (n = 0; n < model->num_terms; n++){
    	model->prob0_w[n] = emvars->prob0_w_temp[n];
		for (j = 0; j< model->num_topics; j++){
			emvars->wbar[j][n] += BIAS;
			emvars->mu[j] += (1-model->u[j][n])*emvars->prob0_w_temp[n];
		}
    }
    for (j = 0; j < model->num_topics; j++){
    	for (n = 0; n < model->num_terms; n++){
    		emvars->wbar_sum[j] += model->u[j][n]*BIAS;
		}
    	if (emvars->mu[j] != 1){
    		emvars->mu[j] = emvars->wbar_sum[j]/(1-emvars->mu[j]);
    	}
    	else{
    		printf("topic %d is not used\n",j);
    		continue;
    	}
    	for (n = 0; n < model->num_terms; n++){
    		model->prob_w[j][n] = emvars->wbar[j][n]*model->u[j][n]/emvars->mu[j];
    	}
    }

    //***************************************update switches
    if (switchupdate==0)
    	return;

    //%%%%%%%%%%%%%%%%%%%%%%%  Ujn

   	counter = 0;
   	sum_nj = 0.0;
   	for (jj = 0; jj < model->num_topics; jj++){
   		emvars->ldvjd_sum[jj] = 0.0;
   	}
   	for (jj = 0; jj < model->num_topics; jj++){
   		emvars->t1_sum[jj] = 0;
   		for (dd = 0; dd <c->num_docs; dd++){
   			emvars->ldvjd_sum[jj] += c->docs[dd].total*alpha->v[dd][jj];
   		}
   		for (nn = 0; nn < model->num_terms; nn++){
   			emvars->t1_sum[jj] += model->u[jj][nn];
		}
   		sum_nj += emvars->t1_sum[jj];
   	}

   	p0t = 0.0;
   	while (counter < 20){
   		change = 0;
		random_permute(model->num_topics, emvars->tpcpermute);
   		for (j0 = 0; j0 < model->num_topics; j0++){
			j = emvars->tpcpermute[j0];
			random_permute(model->num_terms, emvars->wrdpermute);
   			for (n0 = 0; n0 < model->num_terms; n0++){
				n = emvars->wrdpermute[n0];

   				current = model->u[j][n];
   				model->u[j][n] = 1-current; //temp. flip this switch

   				sum_nj = sum_nj - current + 1-current;
   				emvars->t1_sum[j] = emvars->t1_sum[j] - current + 1-current;
   				if (emvars->t1_sum[j] == 0){
   					printf("topic %d is never used2\n",j);
   					continue;
   				}
   				prev_mu = emvars->mu[j];
   				prev_wbar_sum = emvars->wbar_sum[j];
   				temp_prob = emvars->prob0_w_temp[n];

   				if (emvars->mu[j] == 1)
   					temp1 = 1;
   				else
   					temp1 = emvars->wbar_sum[j]/emvars->mu[j];
   				if (current == 1){
   					temp1 -= emvars->prob0_w_temp[n];
   					emvars->wbar_sum[j] -= emvars->wbar[j][n];
   				}
   				else{
   					temp1 += emvars->prob0_w_temp[n];
   					emvars->wbar_sum[j] += emvars->wbar[j][n];
   				}

   				if (temp1 != 1){
   					emvars->mu[j] = emvars->wbar_sum[j]/(temp1);
   				}
   				else{
   					printf("topic %d is not used3\n",j);
   					emvars->prob0_w_temp[n] = temp_prob;
   					emvars->mu[j] = prev_mu;
   					emvars->wbar_sum[j] = prev_wbar_sum;
   					model->u[j][n] = current;
   					continue;
   				}


   				//************** delta cost

   				t3 = log(emvars->ldvjd_sum[j]/2.0/pi);
   				nn=n;
   				dcost = 0.5*t3;
   				if (current==1)	dcost=-dcost;

				t4 = 0.0; // comment out this part for using the exact form (for the image dataset in the paper)
				p0t = sum_nj/model->num_terms/model->num_topics;
				h = -p0t*log(p0t+1e-15)-(1-p0t)*log(1-p0t+1e-15);
   				t4 = h*log(2.0)*model->num_terms;
   				tsum = sum_nj-(1-current) + current;
				p0t = tsum/model->num_terms/model->num_topics;;
				h = -p0t*log(p0t+1e-15)-(1-p0t)*log(1-p0t+1e-15);
   				t4 -= h*log(2.0)*model->num_terms;
   				dcost += t4;

   				//**************** delta lkh
   				dl = 0;
   				if (current == 0)
   					dl = emvars->wbar[j][n]*(log(emvars->wbar[j][n])-log(emvars->prob0_w_temp[n]))
   						-log(emvars->mu[j]/prev_mu)*emvars->wbar_sum[j]-emvars->wbar[j][n]*log(prev_mu);
   				else
   					dl = -emvars->wbar[j][n]*(log(emvars->wbar[j][n])-log(emvars->prob0_w_temp[n]))
   						-log(emvars->mu[j]/prev_mu)*emvars->wbar_sum[j]+emvars->wbar[j][n]*log(prev_mu);

   				dbic = dcost-dl;
   				if (dbic<0){
   					model->u[j][n]=1-current;
   					change++;
   				}
   				else{
   					emvars->t1_sum[j] = emvars->t1_sum[j]-(1-current) + current;
   					sum_nj = sum_nj -(1-current) + current;
   					model->u[j][n] = current;
   					emvars->prob0_w_temp[n] = temp_prob;
   					emvars->mu[j] = prev_mu;
   					emvars->wbar_sum[j] = prev_wbar_sum;
   				}
   			}
   		}
   		printf("%d u update(s)\n",change);
   		counter++;
   		if (change == 0)
   			break;
   		else
   			change = 0;
   	}

   	for (n = 0; n < model->num_terms; n++){
   		model->prob0_w[n] = emvars->prob0_w_temp[n];
   		for (j = 0; j < model->num_topics; j++)
   			model->prob_w[j][n] = emvars->wbar[j][n]*model->u[j][n]/emvars->mu[j];
   	}


    //%%%%%%%%%%%%%%%%%%%%%%%  Vjd update
   	for (jj = 0; jj < model->num_topics; jj++){
   		emvars->ldvjd_sum[jj] = 0.0;
	}
	for (jj = 0; jj < model->num_topics; jj++){
		emvars->gamma1[jj] = 0.0;
		emvars->gamma2[jj] = 0.0;
		emvars->sum_vjd2[jj] = 0.0;
		for (dd = 0; dd < c->num_docs; dd++){
			emvars->ldvjd_sum[jj] += c->docs[dd].total*alpha->v[dd][jj];
			emvars->sum_vjd2[jj] += alpha->v[dd][jj];
		}
	}
	for (n = 0; n < model->num_terms; n++){
		sum_gamma2 = 0.0;
		for (jj = 0; jj < model->num_topics; jj++){
			emvars->gamma1[jj] += model->u[jj][n];
			sum_gamma2 += emvars->ldvjd_sum[jj]*(1-model->u[jj][n]);
		}
		for (jj = 0; jj < model->num_topics; jj++){
			if (sum_gamma2 != 0) emvars->gamma2[jj] += (1-model->u[jj][n])/sum_gamma2;
		}
	}

	sum_md = 0.0;
	for (d = 0; d < c->num_docs; d++){
		emvars->m_d[d] = 0.0;
		for (j = 0; j < model->num_topics; j++){
			emvars->m_d[d] += alpha->v[d][j];
		}
		sum_md += emvars->m_d[d];
	}

   	counter = 0;
   	sum_vjd = 0;

    while (counter < 10){
	change = 0;
	for (d = 0; d < c->num_docs; d++){
		random_permute(model->num_topics, emvars->tpcpermute);
		for (j0 = 0; j0 < model->num_topics; j0++){
			j = emvars->tpcpermute[j0];
			current = alpha->v[d][j];
			t2_sum = 0.0;
			sum_vjd = 0.0;
			//trial update alpha[d][j]
			for (jj = 0; jj < model->num_topics; jj++){
				sum_vjd += alpha->v[d][jj];
				emvars->ldvjd_sum_prev[jj] = emvars->ldvjd_sum[jj];
			}
			if ((sum_vjd == 1) & (alpha->v[d][j] == 1)){ //if this is the only switch on for this doc, skip ...
				continue;
			}
			if ((emvars->sum_vjd2[j] == 1) & (alpha->v[d][j] == 1)) // if this is the only document that uses this topic, skip ...
				continue;

			//trial flip
			alpha->v[d][j] = 1-current;
			alpha_sum = 0.0;
			for (jj = 0; jj < model->num_topics; jj++){
				//alpha_temp[jj] = myrand();
				emvars->alpha_temp[jj] = 0.5;
				alpha_sum += emvars->alpha_temp[jj]*alpha->v[d][jj];
			}
			for (jj = 0; jj < model->num_topics; jj++){
				emvars->alpha_temp[jj] = emvars->alpha_temp[jj]*alpha->v[d][jj]/alpha_sum;
				emvars->alpha_prev[jj] = emvars->alpha_temp[jj];
				emvars->alpha_temp[jj] = 0.0;
			}
			tcnt = 0;
			while(tcnt < 2){
				alpha_sum = 0.0;
				for (n = 0; n < c->docs[d].length; n++){	//trial-update alpha
					sum_gamma = 0;
					for (jj = 0; jj < model->num_topics; jj++){
						if (alpha->v[d][jj] == 0){
							emvars->gamma[jj] = 0.0;
							continue;
						}
						if (model->u[jj][c->docs[d].words[n]] == 0)
							emvars->gamma[jj] = alpha->v[d][jj]*emvars->alpha_prev[jj]*model->prob0_w[c->docs[d].words[n]];
						else
							emvars->gamma[jj] = alpha->v[d][jj]*emvars->alpha_prev[jj]*model->prob_w[jj][c->docs[d].words[n]];
						sum_gamma += emvars->gamma[jj];
					}
					if (sum_gamma == 0)
						continue;
					for (jj = 0; jj < model->num_topics; jj++){
						emvars->gamma[jj] = emvars->gamma[jj]/sum_gamma;
						emvars->alpha_temp[jj] += emvars->gamma[jj]*c->docs[d].counts[n]*alpha->v[d][jj];
						alpha_sum += emvars->gamma[jj]*c->docs[d].counts[n]*alpha->v[d][jj];
					}
				}
				if (alpha_sum == 0)
					continue;
				for (jj = 0; jj < model->num_topics; jj++){
					emvars->alpha_temp[jj] = emvars->alpha_temp[jj]/alpha_sum;
					emvars->alpha_prev[jj] = emvars->alpha_temp[jj];
					emvars->alpha_temp[jj] = 0.0;
				}
				tcnt++;
			}
			for (jj = 0; jj < model->num_topics; jj++){
				emvars->alpha_temp[jj] = emvars->alpha_prev[jj];
			}

			//************** delta bic
			emvars->ldvjd_sum[j] -= c->docs[d].total*current;
			emvars->ldvjd_sum[j] += c->docs[d].total*(1-current);
			t4 = 0;
			t3 = 0;

			t3 = emvars->gamma1[j]*log(emvars->ldvjd_sum[j]/emvars->ldvjd_sum_prev[j]);
			t4 = (1-current-current)*emvars->gamma2[j];
			//reverse trial flip
			alpha->v[d][j] = current;
			t2 = log(((double) c->docs[d].total)/2.0/pi);
			dcost = 0.5*t2;
			if (current==1)
				dcost=-dcost;
			dcost += 0.5*t3;
			t1=0.0;
			t1 -= lgamma(model->num_topics+1.0)-lgamma(sum_vjd+1.0)-lgamma(model->num_topics-sum_vjd+1.0);
			tsum_vjd = sum_vjd - current + 1-current;
			t1 += lgamma(model->num_topics+1.0)-lgamma(tsum_vjd+1.0)-lgamma(model->num_topics-tsum_vjd+1.0);
			dcost += t1;

			dl = 0.0;
			for (n = 0; n < c->docs[d].length; n++){
				temp1 = 0.0;
				temp2 = 0.0;
				for(jj = 0; jj < model->num_topics; jj++){
					if (jj == j){
						if ((model->u[jj][c->docs[d].words[n]] == 0)){
							temp1 += current*alpha->alpha[d][jj]*model->prob0_w[c->docs[d].words[n]];
							temp2 += (1-current)*emvars->alpha_temp[jj]*model->prob0_w[c->docs[d].words[n]];
						}
						else{
							temp1 += current*alpha->alpha[d][jj]*model->prob_w[jj][c->docs[d].words[n]];
							temp2 += (1-current)*emvars->alpha_temp[jj]*model->prob_w[jj][c->docs[d].words[n]];
						}
					}
					else if(alpha->v[d][jj] != 0){
						if ((model->u[jj][c->docs[d].words[n]] == 0)){
							temp1 += alpha->v[d][jj]*alpha->alpha[d][jj]*model->prob0_w[c->docs[d].words[n]];
							temp2 += alpha->v[d][jj]*emvars->alpha_temp[jj]*model->prob0_w[c->docs[d].words[n]];
						}
						else{
							temp1 += alpha->v[d][jj]*alpha->alpha[d][jj]*model->prob_w[jj][c->docs[d].words[n]];
							temp2 += alpha->v[d][jj]*emvars->alpha_temp[jj]*model->prob_w[jj][c->docs[d].words[n]];
						}
					}
				}
				dl += c->docs[d].counts[n]*log(temp2/temp1);
			}
			dbic = dcost-dl;
			if (dbic < 0){
				alpha->v[d][j] = 1-current;
				change++;
				emvars->sum_vjd2[j] = emvars->sum_vjd2[j]-current+(1-current);
				for (jj = 0; jj < model->num_topics; jj++){
					alpha->alpha[d][jj] = emvars->alpha_temp[jj];
				}
				emvars->m_d[d] = emvars->m_d[d]-current+1-current;
				sum_md = sum_md -current+1-current;
			}
			else{
				alpha->v[d][j] = current;
				emvars->ldvjd_sum[j] = emvars->ldvjd_sum_prev[j];
			}
		}
	}
	counter++;
	printf("%d v update(s)\n",change);
	if (change == 0)
		break;
	else
		change = 0;
   }

}



double compute_likelihood(ptm_alpha* alpha, ptm_model* model, corpus* corpus, double* likelihood)
{
	int j, d, n, jj;
	double t1, t2, t3,t4;
	double sw, sd, cost;
	double Md , Nj;
	double lbar_j,sum_nj,p0,h;
	double *nj;
	nj = malloc(sizeof(double)*model->num_topics);

	//computing likelihood

	*likelihood = 0.0;
	for (d = 0; d < corpus->num_docs; d++){
		sd = 0.0;
		for (n = 0; n < corpus->docs[d].length; n++){
			sw = 0.0;
			for (j = 0; j < model->num_topics; j++){
				if (model->u[j][corpus->docs[d].words[n]]==0)
					sw += (alpha->alpha[d][j]) * alpha->v[d][j] *model->prob0_w[corpus->docs[d].words[n]];
				else
					sw += (alpha->alpha[d][j]) * alpha->v[d][j] *model->prob_w[j][corpus->docs[d].words[n]];
			}
			if (sw == 0){
				for (jj=0; jj<model->num_topics; jj++)
					printf("%d %5.5lf, %d, %d, %5.10lf, %5.5lf\n",n, alpha->alpha[d][jj],alpha->v[d][jj],model->u[jj][corpus->docs[d].words[n]],model->prob_w[jj][corpus->docs[d].words[n]],model->prob0_w[corpus->docs[d].words[n]]);
				printf("zero prob ");}
			else
				sd += log(sw)*corpus->docs[d].counts[n];
		}
		*likelihood += sd;
	}

	//computing cost terms

	t1 = 0.0;
	t2 = 0.0;
	sum_nj = 0.0;
	for (j = 0; j < model->num_topics; j++){
		lbar_j = 0.0;
		for (d = 0; d < corpus->num_docs; d++){
			lbar_j += alpha->v[d][j]*corpus->docs[d].total;
		}
		Nj = 0.0;
		for (n = 0; n < model->num_terms; n++){
			if (model->u[j][n]==1){
				Nj += 1.0;
			}
		}
		nj[j] = Nj;
		sum_nj += Nj;
		t1 += 0.5*Nj*log(lbar_j/2/pi);
	}

	p0 = sum_nj/model->num_terms/model->num_topics;
	h = -p0*log(p0+1e-15)-(1-p0)*log(1.0-p0+1e-15);
	t2 = model->num_topics*h*model->num_terms*log(2.0)-0.5*log(model->num_terms*model->num_topics);
    //t2 = model->num_topics*model->num_terms*log(2.0); //using the exact form (for the image dataset)

	t3 = 0.0;
	t4 = 0.0;
	for (d = 0; d < corpus->num_docs; d++){
		Md = 0.0;
		for (j = 0; j < model->num_topics; j++){
			if (alpha->v[d][j]==1){
				Md += 1.0;
			}
		}
		t3 += 0.5*(Md-1)*log(((double)corpus->docs[d].total)/2.0/pi);
		t4 += log(model->num_topics)+lgamma(model->num_topics+1.0)-lgamma(Md+1.0)-lgamma(model->num_topics-Md+1.0);
	}

	cost = t1 + t3;
	cost += t2 + t4;

	free(nj);
	return(cost);

}



void inference(char* model_root, char* directory, corpus* corpus)
{
    FILE* fileptr;
    char filename[100];
    int i, d, j;
    double likelihood, likelihood_old = 0, converged = 1;
	double *alpha_temp, *gamma;
	double cost;
	char* save = "test";
    ptm_model *model;
    ptm_alpha* alpha = NULL;

    model = load_ptm_model(model_root);
    alpha = new_ptm_alpha(corpus->num_docs, model->num_topics);
    random_initialize_alpha_test(model, alpha, corpus);

    alpha_temp = malloc(sizeof(double)*model->num_topics);
    gamma = malloc(sizeof(double)*model->num_topics);

    sprintf(filename, "%s/%s-lhood.dat", directory,save);
    fileptr = fopen(filename, "w");

    cost = compute_likelihood(alpha, model, corpus, &likelihood);
    likelihood = cost - likelihood;
    likelihood_old = likelihood;
    i = 0;
    while (((converged < 0) || (converged > EM_CONVERGED) || (i <= 2)) && (i <= EM_MAX_ITER))
	{
		i++;
		printf("**** em iteration %d ****\n", i);

		em_infer(corpus, model, alpha, alpha_temp, gamma);

		cost=compute_likelihood(alpha, model, corpus, &likelihood);

		// check for convergence
		converged = (likelihood_old - likelihood) / (likelihood_old);
		likelihood_old = likelihood;
		likelihood = cost - likelihood;

		// save likelihood
		fprintf(fileptr, "%10.10f\t%10.10f\t%5.5e\n", likelihood, cost-likelihood, converged); //check if this is the last updated likelihood
		fflush(fileptr);

	}

	em_infer(corpus, model, alpha, alpha_temp, gamma);
	cost = compute_likelihood(alpha, model, corpus, &likelihood);
	converged = (likelihood_old - likelihood) / (likelihood_old);
	likelihood = cost - likelihood;

	// save model and likelihood
	fprintf(fileptr, "%10.10f\t%10.10f\t%5.5e\n", likelihood, cost-likelihood, converged);
	fflush(fileptr);
	sprintf(filename, "%s/%s", directory,save);
	save_ptm_alpha(filename, alpha, corpus->num_docs, model->num_topics);

	free(gamma);
	free(alpha_temp);
	//free model, alpha, emvars
	free(model->prob0_w);
	for (j = 0; j < model->num_topics; j++){
		free(model->prob_w[j]);
		free(model->u[j]);
	}
	free(model);
	for (d = 0; d < corpus->num_docs; d++){
		free(alpha->alpha[d]);
		free(alpha->v[d]);
	}
	free(alpha);
}


void em_infer(corpus* c, ptm_model* model, ptm_alpha* alpha, double* alpha_temp, double* gamma)
{
    int n, j, d;
    double t1;
    double alpha_sum=0.0;

    for (j = 0; j < model->num_topics; j++){
    	alpha_temp[j] = 0.0;
    }

    for (d = 0; d < c->num_docs; d++){

    	alpha_sum = 0.0;
    	for (n = 0; n < c->docs[d].length; n++){
    		//compute gamma for this word
    		t1 = 0.0;
    		for (j = 0; j < model->num_topics; j++){
    			if (alpha->v[d][j] == 0){
    				gamma[j] = 0.0;
    				continue;
    			}
    			if (model->u[j][c->docs[d].words[n]] == 0)
    				gamma[j] = alpha->v[d][j]*alpha->alpha[d][j]*model->prob0_w[c->docs[d].words[n]];
    			else
    				gamma[j] = alpha->v[d][j]*alpha->alpha[d][j]*model->prob_w[j][c->docs[d].words[n]];
    			t1 += gamma[j];
    		}
    		if (t1 == 0){
    			printf("%5.10lf, %5.10lf, %d ,%d, %5.10lf\n",model->prob_w[j][c->docs[d].words[n]],model->prob0_w[c->docs[d].words[n]],model->u[j][c->docs[d].words[n]],alpha->v[d][j],alpha->alpha[d][j]);
    		}
			for (j = 0; j < model->num_topics; j++){
				gamma[j] = gamma[j]/t1;
				alpha_temp[j] += gamma[j]*c->docs[d].counts[n]*alpha->v[d][j];
				alpha_sum += gamma[j]*c->docs[d].counts[n]*alpha->v[d][j];
			}
    	}
		for (j = 0; j < model->num_topics; j++){
			alpha_temp[j] = alpha_temp[j]/alpha_sum;
			alpha->alpha[d][j] = alpha_temp[j];
			alpha_temp[j] = 0.0;
		}
    }

}


corpus* read_data(char* data_filename)
{
    FILE *fileptr;
    int length, count, word, n, nd, nw;
    corpus* c;

    printf("reading data from %s\n", data_filename);
    c = malloc(sizeof(corpus));
    c->docs = 0;
    c->num_terms = 0;
    c->num_docs = 0;
    fileptr = fopen(data_filename, "r");
    nd = 0; nw = 0;
    while ((fscanf(fileptr, "%10d", &length) != EOF)){
		c->docs = (document*) realloc(c->docs, sizeof(document)*(nd+1));
		c->docs[nd].length = length;
		c->docs[nd].total = 0;
		c->docs[nd].words = malloc(sizeof(int)*length);
		c->docs[nd].counts = malloc(sizeof(int)*length);
		for (n = 0; n < length; n++){
			fscanf(fileptr, "%10d:%10d", &word, &count);
			//word = word - OFFSET;
			c->docs[nd].words[n] = word;
			c->docs[nd].counts[n] = count;
			c->docs[nd].total += count;
			if (word >= nw) { nw = word + 1; }
		}
		nd++;
    }
    fclose(fileptr);
    c->num_docs = nd;
    c->num_terms = nw;
    printf("number of docs    : %d\n", nd);
    printf("number of terms   : %d\n", nw);
    return(c);
}

void write_wrd_asgnmnts(char* filename, corpus* c, ptm_model* model, ptm_alpha* alpha, ptm_emvars* emvars){

	int n, d, j, jmax;
	double t1, maxalpha;
	FILE* fileptr;

	fileptr = fopen(filename, "w+");
	for (d = 0; d < c->num_docs; d++){
		fprintf(fileptr, "%03d", c->docs[d].length);
		for (n = 0; n < c->docs[d].length; n++){
			//compute Pz(gamma) for this word
			t1 = 0;
			maxalpha = -1;
			jmax = 0;
			for (j = 0; j < model->num_topics; j++){
				if (alpha->v[d][j] == 0){
					emvars->gamma[j] = 0.0;
					continue;
				}
				if (model->u[j][c->docs[d].words[n]] == 0)
					emvars->gamma[j] = alpha->v[d][j]*alpha->alpha[d][j]*model->prob0_w[c->docs[d].words[n]];
				else
					emvars->gamma[j] = alpha->v[d][j]*alpha->alpha[d][j]*model->prob_w[j][c->docs[d].words[n]];
				t1 += emvars->gamma[j];
			}
			if (t1==0){
				printf("%5.10lf, %5.10lf, %d ,%d, %5.10lf\n",model->prob_w[j][c->docs[d].words[n]],model->prob0_w[c->docs[d].words[n]],model->u[j][c->docs[d].words[n]],alpha->v[d][j],alpha->alpha[d][j]);
			}
			for (j = 0; j < model->num_topics; j++){
				if (emvars->gamma[j] == 0)
					continue;
				emvars->gamma[j] = emvars->gamma[j]/t1;
				if (emvars->gamma[j] > maxalpha){
					maxalpha = emvars->gamma[j];
					jmax = j;
				}
			}
			fprintf(fileptr, " %04d:%02d", c->docs[d].words[n], jmax);
		}
		fprintf(fileptr, "\n");
	}
}



void random_permute(int size, int* vec){

	int i, j, temp;
	
	for (j = 0; j < size; j++){
		vec[j] = j;
	}
	for (i = size-1; i > 0; i--){
		j = ((int)(size*myrand()))%(i+1);
		temp = vec[j];
		vec[j] = vec[i];
		vec[i] = temp;
	}

}
