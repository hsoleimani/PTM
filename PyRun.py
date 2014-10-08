# Python code to reproduce experiments in the paper
import numpy as np
import funcs
import os

trainingfile = '' #training data set
testfile = '' #test data set
obs_tfile = '' #Observed data set
ho_tfile = '' # held-out data set
tr_label_file = '' #training class labels
t_label_file = '' #test class labels
D =  # num of training documents
C =  # num of class labels
N =  # num of words
Mmax =  # Max num of topics
Mmin =  # Min num of topics
step =  # Topic reduction step size

refile = open('results.dat', 'w+')
refile.write('M, bic, Etr_lkh, ho_lkh, avg_tpc/doc, avg_wrd/tpc, num_unq_wrds, tr_ccr, t_ccr, runtime, ')
refile.write('uv_avg_tpc/doc, uv_avg_wrd/tpc, uv_num_unq_wrds\n')
refile.close()

k = reversed(range(Mmin,Mmax+step,step))
for M in k:
    path = 'dir' + str(M)
    ## run PTM on training data
    cmdtxt = './ptm --num_topics ' + str(M) + ' --corpus ' + trainingfile + ' --convergence 1e-4'
    if M == Mmax:
        cmdtxt = cmdtxt + ' --init seeded --dir ' + path
    else:
        cmdtxt = cmdtxt + ' --init load --model ' + path + '/init --dir ' + path 
    os.system(cmdtxt)

    ### read training topic proportions
    theta = np.loadtxt(path+'/final.alpha')
    vswitch = np.loadtxt(path+'/final.v') 
    #read word probabilities
    beta = np.exp(np.loadtxt(path+'/final.beta'))
    uswitch = np.loadtxt(path+'/final.u') 
    for j in range(M):
        beta[uswitch[:,j]==1, j] = beta[uswitch[:,j]==1, 0]
   
    
    # compute sparsity measures
    (avg_tpcs, avg_wrds, unq_wrds) = funcs.topic_word_sparsity(path+'/word-assignments.dat',N,M,uswitch)
    (uv_avg_tpcs, uv_avg_wrds, uv_unq_wrds) = funcs.switch_topic_word_sparsity(uswitch,vswitch,N,M)
    
    # read training likelihood
    lk = np.loadtxt(path+'/likelihood.dat')
    bic = lk[-1,0]
    lkh = lk[-1,1]
    runtime = lk[-1,3]

    # inference on test set
    cmdtxt = './ptm --task test' + ' --corpus ' + testfile + ' --convergence 1e-4'
    cmdtxt = cmdtxt + ' --dir ' + path + ' --model ' + path + '/final' 
    os.system(cmdtxt) 
      
    ## compute likelihood on held-out set
    Etrlk = funcs.compute_lkh(trainingfile, beta[:,1:M+1], theta)

    ### read test topic proportions
    theta_test = np.loadtxt(path+'/test.alpha')
    
    ## measure class label consistency
    (ccr_tr,tpc_lbl_distn) = funcs.classifier_training(tr_label_file,theta,C,M)
    ccr_t = funcs.classifier_test(t_label_file,tpc_lbl_distn,theta_test)
    
    ## inference on observed test set
    cmdtxt = './ptm --task test' + ' --corpus ' + obs_tfile + ' --convergence 1e-4'
    cmdtxt = cmdtxt + ' --dir ' + path + ' --model ' + path + '/final' 
    os.system(cmdtxt) 
    
    ## topic proportions of the observed set
    theta_test = np.loadtxt(path+'/test.alpha')
    
    ## compute likelihood on held-out set
    ho_lkh = funcs.compute_lkh(ho_tfile, beta[:,1:M+1], theta_test)
    
    ## prepare for the next model order (writes it in the next folder)
    if  (M >= (Mmin+step)):
        next_path = 'dir' + str(M-step)
        os.system('mkdir -p '+next_path)
        funcs.prepare_next_forptm(step, path, next_path, theta)
    
    ## save useful stuff
    # results file
    refile = open('results.dat', 'a')
    refile.write(str(M) + ', ' + str(bic) + ', ' + str(Etrlk) + ', ' + str(ho_lkh) + ', ' + str(avg_tpcs) + ', ')
    refile.write(str(np.mean(avg_wrds)) + ', ' + str(np.sum(unq_wrds)) + ', ' + str(ccr_tr) + ', ' + str(ccr_t) + ', ')
    refile.write(str(runtime)+', '+str(uv_avg_tpcs) + ', ' + str(np.mean(uv_avg_wrds)) + ', ' + str(np.sum(uv_unq_wrds)) + '\n')
    refile.close()
    
    ## delete other files
    os.system('rm -rf '+path)
    
