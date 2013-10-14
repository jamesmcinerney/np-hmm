'''
Created on 3 Oct 2013

@author: James McInerney
'''


from numpy import *
from matplotlib.pyplot import *
from numpy.linalg.linalg import inv, det
from scipy.special.basic import digamma
import time
#from util.viz import create_cov_ellipse
from scipy.stats import vonmises
import sys
from model.util import lnNorm, log0
from model.sensors import MVGaussianSensor
import itertools as itt
from matplotlib.pyplot import *
from testing.util import create_cov_ellipse
from model.viterbi import viterbiLog


ROOT = '/media/8A8823BF8823A921/Dropbox/variational/'

set_printoptions(threshold=nan,suppress=1)
#seterr(all='raise')


def infer(N,X_all,K,sensors,thres=1e-4,max_itr=200,min_itr=10,stateHyperparams=None,
          VERBOSE=0,Z_grnd=None,useMix=0,useHMM=1,plotSensor=None,plotX=None,
          mu_grnd=None):
    #X_all: list of observations (in same ordering as |sensors|)
    #K: truncation parameter (for number of components explaining data)

    #define hyperparameters by default:
    if stateHyperparams is None:
        stateHyperparams = {'alpha_pi':1.0, #hyperparam for initial state DP
                       'alpha_a':1.0, #hyperparam for transition DP
                       } 

    alpha_pi = stateHyperparams['alpha_pi'] #hyperparam for initial state DP
    alpha_a  = stateHyperparams['alpha_a'] #hyperparam for transition DP
    
    #randomly initialise expected values of random variables:
    exp_s = array([random.uniform(0,100,(K,K)) for _ in range(N)])
    for n in range(N): exp_s[n,:,:] = exp_s[n,:,:] / exp_s[n,:,:].sum()
    
    #either used ground truth provided for exp_z (for debug purposes) or initialise randomly:
    if Z_grnd is None: 
        exp_z = array([random.dirichlet(ones(K)) for _ in range(N)])
#        exp_z = zeros((N,K)) #
#        for n in range(N): 
#            exp_z[n,0] = 0.99
#            exp_z[n,1:] = 0.01/float(K-1)
    else:
        exp_z = zeros((N,K))
        (N1,KG) = shape(Z_grnd)
        exp_z[:,:KG] = Z_grnd #make Z_grnd shape match exp-z shape

    #rand init of variational parameters:
    #[s.m(X_all[i],exp_z,randInit=1) for (s,i) in zip(sensors,itt.count())]
    tau_pi0,tau_pi1 = ones(K), ones(K)
    tau_a0,tau_a1 = ones((K,K)), ones((K,K))
    tau_ck = ones(K)
    
    if plotSensor is not None:
        #animation:
        ion()    
        fig = figure(figsize=(10,10))
        ax_spatial = fig.add_subplot(1,1,1) #http://stackoverflow.com/questions/3584805/in-matplotlib-what-does-111-means-in-fig-add-subplot111
        circs = []
        ellipseColor = 'r'

    itr = 0
    diff,prev_ln_obs_lik = 1,zeros((N,K)) #stop when parameters have converged (local optimum)
    while (itr<min_itr) or (itr<max_itr and diff>thres):                
        #---------------
        # M-step:
        #---------------
        
        #variational parameters governing latent states:
        if useMix: 
            tau_pi0,tau_pi1 = mixMPi(alpha_pi,exp_z,K)
        else: 
            tau_pi0,tau_pi1 = mPi(alpha_pi,exp_z,K)
            tau_a0,tau_a1 = mA(alpha_a,exp_s,K)
        #tau_ck = alpha_pi + exp_z.sum(axis=0)
        #optimise variational parameters governing observation likelihoods:
        [s.m(X_all[i],exp_z) for (s,i) in zip(sensors,itt.count())]

        
        #---------------
        # E-step:
        #---------------
        
        #calculate observation likelihood of data for each sensor (combined):
        ln_obs_lik = array([s.loglik(X_all[i]) for (s,i) in zip(sensors,itt.count())]).sum(axis=0)
        #print 'ln_obs_lik',ln_obs_lik
        exp_ln_pi = ePi(tau_pi0,tau_pi1,K)
        #exp_ln_pi = digamma(tau_ck) - digamma(tau_ck.sum())
        #find expected values of latent variables:
        if useMix: 
            exp_z = mixEZ(ln_obs_lik, exp_ln_pi, N, K) #mixture model estimation of Z
        else: 
            exp_ln_a = eA(tau_a0,tau_a1,K)
            ln_alpha_exp_z = eFowardsZ(exp_ln_pi, exp_ln_a, ln_obs_lik, N, K) #FORWARDS PASS
            ln_beta_exp_z = eBackwardsZ(exp_ln_pi, exp_ln_a, ln_obs_lik, N, K) #BACKWARDS PASS
            exp_z = eZ(ln_alpha_exp_z, ln_beta_exp_z, N) #find expected state for each time step
            exp_s = eS(exp_ln_a, ln_alpha_exp_z, ln_beta_exp_z, ln_obs_lik, N, K) #find expected transition for each time step
        
        
        
        diff = abs(ln_obs_lik - prev_ln_obs_lik).sum()/float(N*K) #average difference in previous expected value of transition matrix
        prev_ln_obs_lik = ln_obs_lik.copy()
        
        print 'itr,diff',itr,diff
        if VERBOSE:
            lim = 5
            print 'exp_z:\n',exp_z.argmax(axis=1)
            #print 'ln_obs_lik:\n',ln_obs_lik[:lim,:]
            #print 'ln_alpha_exp_z',ln_alpha_exp_z
            #print 'ln_beta_exp_z',ln_beta_exp_z
            
            
        if plotSensor is not None:
            (ks,) = where(exp_z.sum(axis=0)>1.) #only look at active components
            X = plotX
            mvgSensor = plotSensor
            if itr==0:
                sctX = scatter(X[:,0],X[:,1],marker='x',color='g')
                sctZ = scatter(mvgSensor._m[:,0],mvgSensor._m[:,1],color='r')
                if mu_grnd is not None:
                    (K_grnd,_) = shape(mu_grnd)
                    for k in range(K_grnd): scatter(mu_grnd[k,0],mu_grnd[k,1],color='k',marker='d',s=50) #plot ground truth means
            else:
                #ellipses to show covariance of components
                for circ in circs: circ.remove()
                circs = []
                for k in ks:
                    circ = create_cov_ellipse(mvgSensor._S[k], mvgSensor._m[k,:],color=ellipseColor,alpha=0.3) #calculate params of ellipses (adapted from http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals)
                    circs.append(circ)
                    #add to axes:
                    ax_spatial.add_artist(circ)
                (_,XDim) = shape(X)
                hiddenOffsets = 99999*ones((K,XDim)) #hide non-significant components
                hiddenOffsets[ks,:] = mvgSensor._m[ks,:] 
                sctZ.set_offsets(hiddenOffsets)
            draw()
            if itr==0: time.sleep(10.)
            #time.sleep(0.05)
        #next iteration:
        itr+=1
        
        #determine if we can switch off mix:
        if useMix and useHMM and (itr>=max_itr or diff<=thres):
            itr = 1
            useMix = 0
            diff = inf
            prev_ln_obs_lik = 0
            ellipseColor='y'
            print 'Mixture converged. SWTCHING TO HMM INFERENCE'


    print 'completed inference.'
    
    exp_pi = expPi(tau_pi0, tau_pi1,K)
    exp_a = expA(tau_a0,tau_a1,K)
    if useMix: concMatrix = exp_pi
    else: concMatrix = exp_a
    
    print 'final taupi0',tau_pi0
    print 'final taupi1',tau_pi1
    

    return exp_z,sensors,concMatrix,viterbiLog(ln_obs_lik,exp_a,exp_pi)


#---------------
# M-step calculations
#---------------
                
def mPi(alpha_pi,exp_z,K):
    #alpha_pi: hyperparam for DP prior
    #exp_z: expectation of latent variables (we are only interested at time step 0 here)
    #K: truncation param. for DP
    tau_pi0,tau_pi1 = zeros(K), zeros(K)
    for k in range(K):
        tau_pi0[k] = alpha_pi + exp_z[0,k+1:].sum() #hyperparam for this component NOT explaining the data
        tau_pi1[k] = 1. + exp_z[0,k] #hyperparam for this component explaining the data
        
    return tau_pi0,tau_pi1

                
def mixMPi(alpha_pi,exp_z,K):
    #alpha_pi: hyperparam for DP prior
    #exp_z: expectation of latent variables (we are only interested at time step 0 here)
    #K: truncation param. for DP
    tau_pi0,tau_pi1 = zeros(K), zeros(K)
    for k in range(K):
        #print 'exp_z',exp_z
        tau_pi0[k] = alpha_pi + exp_z[:,k+1:].sum() #hyperparam for this component NOT explaining the data
        tau_pi1[k] = 1. + exp_z[:,k].sum() #hyperparam for this component explaining the data
    return tau_pi0,tau_pi1


def mA(alpha_a,exp_s,K):
    #alpha_a: hyperparam for transition matrix
    #exp_s: expectation of latent variables (transitions)
    #K: truncation param. for DP
    tau_a0,tau_a1 = zeros((K,K)), zeros((K,K))
    for i in range(K):
        for j in range(K):
            tau_a0[i,j] = alpha_a + exp_s[:,i,j+1:].sum() #hyperparam for this component NOT explaining the data
            tau_a1[i,j] = 1. + exp_s[:,i,j].sum() #hyperparam for this component explaining the data
    return tau_a0,tau_a1


#---------------
# E-step calculations
#---------------

def ePi(tau_pi0,tau_pi1,K):
    exp_ln_pi = zeros(K)
    acc = digamma(tau_pi0) - digamma(tau_pi0 + tau_pi1)
    for k in range(K): exp_ln_pi[k] = digamma(tau_pi1[k]) - digamma(tau_pi0[k] + tau_pi1[k]) + acc[:k].sum()
    return exp_ln_pi        

def expPi(tau_pi0,tau_pi1,K):
    exp_pi = zeros((1,K))
    acc = tau_pi0 / (tau_pi0 + tau_pi1)
    for k in range(K): exp_pi[0,k] = (acc[:k].prod()*tau_pi1[k]) / (tau_pi0[k] + tau_pi1[k])
    return exp_pi

def eA(tau_a0,tau_a1,K):
    exp_ln_a = zeros((K,K))
    acc = digamma(tau_a0) - digamma(tau_a0 + tau_a1)
    for i in range(K):
        for j in range(K):
            exp_ln_a[i,j] = digamma(tau_a1[i,j]) - digamma(tau_a0[i,j] + tau_a1[i,j]) + acc[i,:j].sum()
    return exp_ln_a
    
    
def expA(tau_a0,tau_a1,K):
    exp_a = zeros((K,K))
    acc = tau_a0/(tau_a0+tau_a1)
    for i in range(K):
        for j in range(K):
            exp_a[i,j] = (acc[i,:j].prod()*tau_a1[i,j])/(tau_a0[i,j]+tau_a1[i,j]) 
    return exp_a

def eZ(ln_alpha_exp_z, ln_beta_exp_z, N):
    #combine the alpha and beta messages to find the expected value of the latent variables:
    ln_exp_z = ln_alpha_exp_z + ln_beta_exp_z
    
    #exponentiate and normalise:
    ln_exp_z -= reshape(ln_exp_z.max(axis=1), (N,1))
    exp_z = exp(ln_exp_z) / reshape(exp(ln_exp_z).sum(axis=1), (N,1))
    #exp_z = lnNorm(ln_exp_z,axis=1)
    return exp_z
    
def mixEZ(ln_obs_lik, exp_ln_pi, N, K):
    #follow mixture (not a time series):
    ln_exp_z = zeros((N,K))
    for k in range(K):
        ln_exp_z[:,k] = exp_ln_pi[k] + ln_obs_lik[:,k]
    
    #exponentiate and normalise:
    ln_exp_z -= reshape(ln_exp_z.max(axis=1), (N,1))
    exp_z = exp(ln_exp_z) / reshape(exp(ln_exp_z).sum(axis=1), (N,1))
    #exp_z = lnNorm(ln_exp_z,axis=1)
    return exp_z
    
def eFowardsZ(exp_ln_pi,exp_ln_a,ln_obs_lik,N,K):
    ln_alpha_exp_z = zeros((N,K)) - inf
    #initial state distribution:
    #print 'exp_invc',exp_invc
    ln_alpha_exp_z[0,:] = exp_ln_pi + ln_obs_lik[0,:]
    for n in range(1,N):
        for i in range(K): #marginalise over all possible previous states:
            ln_alpha_exp_z[n,:] = logaddexp(ln_alpha_exp_z[n,:], ln_alpha_exp_z[n-1,i]+ exp_ln_a[i,:] + ln_obs_lik[n,:])
    return ln_alpha_exp_z 
    
def eBackwardsZ(exp_ln_pi,exp_ln_a,ln_obs_lik,N,K):
    ln_beta_exp_z = zeros((N,K)) - inf
    #final state distribution:
    ln_beta_exp_z[N-1,:] = zeros(K)
    for n in range(N-2,-1,-1):
        for j in range(K): #marginalise over all possible next states:
            ln_beta_exp_z[n,:] = logaddexp(ln_beta_exp_z[n,:], ln_beta_exp_z[n+1,j] + exp_ln_a[:,j] + ln_obs_lik[n+1,j])
    return ln_beta_exp_z

def eS(exp_ln_a, ln_alpha_exp_z, ln_beta_exp_z, ln_obs_lik, N, K):
    ln_exp_s = zeros((N-1,K,K)) #we only care about the transitions, which is why the length of this var is (N-1)
    exp_s = zeros((N-1,K,K))
    for n in range(N-1):
        for i in range(K):
            ln_exp_s[n,i,:] = ln_alpha_exp_z[n,i] + ln_beta_exp_z[n+1,:] + ln_obs_lik[n+1,:]  + exp_ln_a[i,:]
        ln_exp_s[n,:,:] -= ln_exp_s[n,:,:].max()
        exp_s[n,:,:] = exp(ln_exp_s[n,:,:]) / exp(ln_exp_s[n,:,:]).sum()
    return exp_s

if __name__ == "__main__":
    from testing.run import test1
    test1()