import numpy as np
import re

def classifier_training(tr_label_file,theta,C,M):
    ## read training labels
    tr_label = open(tr_label_file,'r')
    tpc_lbl_distn = np.zeros((M,C))
    
    #learning topic_label_multinomials
    d = 0
    while True:
        labelline = tr_label.readline()
        if len(labelline)==0:
            break
        lbl = int(labelline.split()[0]) # assuming single label per doc -- labeling starts from zero
        tpc_lbl_distn[:,lbl] += theta[d,].T
        
        d += 1  
    
    tpc_lbl_distn = tpc_lbl_distn/(np.tile(np.sum(tpc_lbl_distn,1),(C,1)).T)
    
    # ccr on the training set
    tr_label.seek(0)
    ccr = 0.0
    d = 0
    while True:
        labelline = tr_label.readline()
        if len(labelline)==0:
            break
        lbl = int(labelline.split()[0]) # assuming single label for now/ should start from zero
        dlp = np.dot(theta[d,:],tpc_lbl_distn)
        pred_lbl = np.argmax(dlp)
        if (pred_lbl==lbl):
            ccr += 1.0
        d += 1
        
    ccr = ccr/float(d)
    tr_label.close()
    return(ccr,tpc_lbl_distn)

def classifier_test(label_file,tpc_lbl_distn,theta):
    
    labelfile = open(label_file,'r')
    ccr = 0.0
    d = 0
    while True:
        labelline = labelfile.readline()
        if len(labelline)==0:
            break
        lbl = int(labelline.split()[0]) 
        dlp = np.dot(theta[d,:],tpc_lbl_distn)
        pred_lbl = np.argmax(dlp)
        if (pred_lbl==lbl):
            ccr += 1.0
        d += 1
        
    ccr = ccr/float(d)
    labelfile.close()
    return(ccr)


def topic_word_sparsity(filename,N,M,u):
    # measure topic/word sparsity of LDA
    fp = open(filename,'r')
    
    beta = np.zeros((N,M))
    num_topics = 0.0
    d = 0
    while True:
        doc = fp.readline()
        if len(doc)==0:
            break
        tpcs = re.findall('[0-9]*:([0-9]*)',doc)
        wrds = re.findall('([0-9]*):[0-9]*',doc)
        num_topics += len(set(tpcs))
        for n in range(len(wrds)):
            w = int(wrds[n])
            j = int(tpcs[n]) 
            if (u[w,j]==1):
                beta[w,j] = 1
        d += 1
    
    avg_tpcs = float(num_topics)/float(d)
    avg_wrds = np.sum(beta,0)
    unq_wrds = np.max(beta,1)
    fp.close()
    return(avg_tpcs,avg_wrds,unq_wrds)


def switch_topic_word_sparsity(u,v,N,M):
    # measure topic/word sparsity based on uv switches
    
    avg_tpcs = np.mean(np.sum(v,1))
    unq_wrds = np.max(u,1)
    avg_wrds = np.sum(u,0)
    
    return(avg_tpcs,avg_wrds,unq_wrds)

def compute_lkh(docfile, beta, theta):
    
    eps = 1e-30
    lkh = 0.0
    
    fp = open(docfile,'r')
    d = 0
    while True:
        doc = fp.readline()
        if (len(doc)==0):
            break
        wrds = re.findall('([0-9]*):[0-9]*',doc)
        cnts = re.findall('[0-9]*:([0-9]*)',doc)
        ld = len(wrds)
        for n in range(ld):
            lkh += float(cnts[n])*np.log(np.dot(theta[d,:],beta[int(wrds[n]),:])+eps)
        d += 1
    
    fp.close()
    return(lkh)


def prepare_next_forptm(step,curr_path, next_path, theta):
    
    otherfile = open(curr_path+'/final.other','r')
    nextotherfile = open(next_path+'/init.other','w+')    
    otherline = otherfile.readline()
    M = int(otherline.split()[1])
    nextotherfile.write('num_topics ' + str(M-step) + '\n')
    otherline = otherfile.readline() # reads num_terms
    nextotherfile.write(otherline)
    otherfile.close()
    nextotherfile.close()
    
    sum_theta = np.sum(theta,0)
    lmt = []
    for j in range(step):
        Mmin = np.argmin(sum_theta)
        sum_theta[Mmin] = np.inf
        lmt.append(Mmin)
    
    betafile = open(curr_path+'/final.beta','r')
    nextbetafile = open(next_path+'/init.beta','w+')
    ufile = open(curr_path+'/final.u','r')
    nextufile = open(next_path+'/init.u','w+')
    while True:
        betaline = betafile.readline()
        if len(betaline)==0:
            break
        uline = ufile.readline().split()
        betaline = betaline.split()
        nextbetafile.write(betaline[0]+ ' ')
        for j in range(M):
            if j in lmt:
                continue
            nextbetafile.write(betaline[j+1] + ' ')
            nextufile.write(uline[j] + ' ')
        nextbetafile.write('\n')
        nextufile.write('\n')
    betafile.close()
    nextbetafile.close()
    ufile.close()
    nextufile.close()
