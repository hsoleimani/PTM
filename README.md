Parsimonious Topic Model

For details of the algorithm, please check the paper, Hossein Soleimani and David J. Miller,
 "Parsimonious Topic Models with Salient Word Discovery", arXiv:1401.6169.

(C) Copyright 2014, Hossein Soleimani
		     David J. Miller


 This program is free program; you can redistribute it and/or modify it under the terms of 
 the GNU General Public License as published by the Free Software Foundation; either 
 version 2 of the License, or any later version. This program is distributed in the hope 
 that it will be useful, but WITHOUT ANY WARRANTY; without even he implied warranty of 
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License 
 for more details.


1) Compile the program in Linux-based system. 
type: make 

2) Type "./ptm"

3) Options:

	--task			training/test, (default: training )
	--num_topics	number of topics
	--directory		directory to save the output
	--corpus		corpus file, in lda-c format; i.e. each line is of the form
				[L] [term_1]:[count] ... [term_L]:[count]
				where L is the number of unique terms in the document, and the [count]
				associated with each term is the number of times that term appears in the document.
	--init			initialization method. seeded/random/load
				seeded: see the paper for details of this method
				random: random initialization
				load: load word probabilities and randomly initialize topic proportions
	--model			name of the model to load
	--max_iter		maximum iterations after which we stop the EM algorithm. (default: 100)
	--convergence	If increase in the log-likelihood is less than "convergence", EM is terminated. (default: 5e-3)
	--save_lag		Save the model at every "save_lag" step. (default: -1)
	--step			Number of topics to remove for next steps' initialization. See the paper for model
				order selection. (default 0)
		
 

4) Output format:
	Training phase saves the follwong files in the directory:
		
		final.alpha:		Contains topic proportions, where each line corresponds to
					a document in the format: [alpha_1] [alpha_2] ... [alpha_M]
					where M is the number of topics
		final.v:		Binary switches for topic proportions (i.e. v switches) in the same 
					format as in final.alpha.
		final.beta		Contains M+1 columns and N rows where each row corresponds to a term 
 					(N: total # unique words)
					First column is the shared model, and each of the next M columns indicates 
					probability of words under that topic.
		final.u			Contains u switches in M columns and N rows
		final.other		First row is the number of topics and the second number of terms
		likelihood.dat:		Contains bic, log-likelihood, and convergence values at each iteration of EM.
		nbar.txt:		Indicates total number of topic-specific words at each iteration of EM. 

	Test step saves the follwong files in the directory:
		test-alpha:		Similar to final.alpha.
		test-lhood:		similar to likelihood.dat

