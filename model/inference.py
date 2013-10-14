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


ROOT = '/media/8A8823BF8823A921/Dropbox/variational/'

set_printoptions(threshold=nan,suppress=1)


def infer(X,K,hyperparams=None,VERBOSE=0,Z_grnd=None):
    #X: continuous observations (possibly multi-dimensional)
    #K: truncation parameter (for number of components explaining data)

    (N,XDim) = shape(X)
    
    #hyperparameters:
    if hyperparams is None:
        hyperparams = {'alpha_pi':1.0, #hyperparam for initial state DP
                       'alpha_a':1.0, #hyperparam for transition DP
                       'beta0':(1e-20)*1., #variance of mean (smaller: broader the means)
                       'v0':XDim+1., #degrees of freedom in inverse wishart
                       'm0':zeros(XDim), #prior mean
                       'W0':(1e0)*eye(XDim) #prior cov (bigger: smaller covariance)
                       }

    alpha_pi = hyperparams['alpha_pi'] #hyperparam for initial state DP
    alpha_a  = hyperparams['alpha_a'] #hyperparam for transition DP
    beta0 = hyperparams['beta0'] #variance of mean (smaller: broader the means)
    v0 = hyperparams['v0'] #degrees of freedom in inverse wishart
    m0 = hyperparams['m0'] #prior mean
    W0 = hyperparams['W0'] #prior cov (bigger: smaller covariance)
    
    #randomly initialise expected values of random variables:
    exp_s = array([[random.dirichlet(ones(K)) for _ in range(K)] for _ in range(N)])
    if Z_grnd is None: exp_z = array([random.dirichlet(ones(K)) for _ in range(N)])
    else: exp_z = Z_grnd
    
    itr,max_itr = 0,200
    diff,min_diff,prev_exp_ln_a = 1,1e-3,zeros((K,K)) #stop when parameters have converged (local optimum)
    while itr<max_itr and diff>min_diff:
        
        #---------------
        # M-step:
        #---------------
        
        #variational parameters governing latent states:
        tau_pi0,tau_pi1 = mPi(alpha_pi,exp_z,K)
        tau_a0,tau_a1 = mA(alpha_a,exp_s,K)
        #variational parameters governing Gaussian observation likelihoods:
        NK = exp_z.sum(axis=0)
        vk = v0 + NK + 1.
        xd = mXd(exp_z,X)
        S = mS(exp_z,X,xd,NK)
        betak = beta0 + NK
        m = mM(K,XDim,beta0,m0,NK,xd,betak)
        W = mW(K,W0,xd,NK,m0,XDim,beta0,S)

        #---------------
        # E-step:
        #---------------
        
        exp_ln_pi = ePi(tau_pi0,tau_pi1,K)
        exp_ln_a = eA(tau_a0,tau_a1,K)
        exp_diff_mu = eDiffMu(X,XDim,NK,betak,m,W,xd,vk,N,K) #eqn 10.64 Bishop
        exp_invc = eInvc(W,vk,XDim,K) #eqn 10.65 Bishop
        #find expected values of latent variables:
        ln_alpha_exp_z = eFowardsZ(exp_ln_pi, exp_ln_a, XDim, exp_invc, exp_diff_mu, vk, betak, N, K) #FORWARDS PASS
        ln_beta_exp_z = eBackwardsZ(exp_ln_pi,exp_ln_a,XDim,exp_invc,exp_diff_mu,vk,betak,N,K) #BACKWARDS PASS
        if Z_grnd is None: 
            exp_z = eZ(ln_alpha_exp_z, ln_beta_exp_z, N)
#            exp_z = ln_alpha_exp_z
#            exp_z -= reshape(exp_z.max(axis=1),(N,1))
#            exp_z = exp(exp_z) / reshape(exp(exp_z).sum(axis=1),(N,1))
        exp_s = eS(exp_ln_a, ln_alpha_exp_z, ln_beta_exp_z, (XDim,exp_invc,exp_diff_mu,vk,betak), N, K)
#        exp_s = zeros((N,K,K))
#        for n in range(N-1): exp_s[n,exp_z[n,:].argmax(),exp_z[n+1,:].argmax()] += 1
        
        
        itr+=1
        diff = abs(exp_ln_a - prev_exp_ln_a).sum()/float(K**2) #average difference in previous expected value of transition matrix
        prev_exp_ln_a = exp_ln_a.copy()
        
        print 'itr,diff',itr,diff
        if VERBOSE:
            print 'exp_z',exp_z.argmax(axis=1)
            #print 'ln_alpha_exp_z',ln_alpha_exp_z
            #print 'ln_beta_exp_z',ln_beta_exp_z
            
            

    print 'completed inference.'
    

    return exp_z,m,S,expA(tau_a0,tau_a1,K)


#---------------
# M-step calculations
#---------------
                
def mPi(alpha_pi,exp_z,K):
    #alpha_pi: hyperparam for DP prior
    #exp_z: expectation of latent variables (we are only interested at time step 0 here)
    #K: truncation param. for DP
    tau_pi0,tau_pi1 = zeros(K), zeros(K)
    for k in range(K):
        #print 'exp_z',exp_z
        tau_pi0[k] = alpha_pi + exp_z[0,k+1:].sum() #hyperparam for this component NOT explaining the data
        tau_pi1[k] = 1. + exp_z[0,k] #hyperparam for this component explaining the data
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

def mXd(Z,X):
    #weighted means (by component responsibilites)
    (N,XDim) = shape(X)
    (N1,K) = shape(Z)
    NK = Z.sum(axis=0)
    assert N==N1
    xd = zeros((K,XDim))
    for n in range(N):
        for k in range(K):
            xd[k,:] += Z[n,k]*X[n,:]
    #safe divide:
    for k in range(K):
        if NK[k]>0: xd[k,:] = xd[k,:]/NK[k]
    
    return xd

def mS(Z,X,xd,NK):
    (N,K)=shape(Z)
    (N1,XDim)=shape(X)
    assert N==N1
    
    S = [zeros((XDim,XDim)) for _ in range(K)]
    for n in range(N):
        for k in range(K):
            B0 = reshape(X[n,:]-xd[k,:], (XDim,1))
            L = dot(B0,B0.T)
            assert shape(L)==shape(S[k]),shape(L)
            S[k] += Z[n,k]*L
    #safe divide:
    for k in range(K):
        if NK[k]>0: S[k] = S[k]/NK[k]
    return S

def mW(K,W0,xd,NK,m0,XDim,beta0,S):
    Winv = [None for _ in range(K)]
    for k in range(K): 
        Winv[k]  = inv(W0) + NK[k]*S[k]
        Q0 = reshape(xd[k,:] - m0, (XDim,1))
        q = dot(Q0,Q0.T)
        Winv[k] += (beta0*NK[k] / (beta0 + NK[k]) ) * q
        assert shape(q)==(XDim,XDim)
    W = []
    for k in range(K):
        try:
            W.append(inv(Winv[k]))
        except linalg.linalg.LinAlgError:
            #print 'Winv[%i]'%k, Winv[k]
            raise linalg.linalg.LinAlgError()
    return W

def mM(K,XDim,beta0,m0,NK,xd,betak):
    m = zeros((K,XDim))
    for k in range(K): m[k,:] = (beta0*m0 + NK[k]*xd[k,:]) / betak[k]
    return m    

#---------------
# E-step calculations
#---------------

def eInvc(W,vk,XDim,K):
    invc = [None for _ in range(K)]
    for k in range(K):
        dW = det(W[k])
        #print 'dW',dW
        if dW>1e-30: ld = log(dW)
        else: ld = 0.0
        invc[k] = sum([digamma((vk[k]+1-i) / 2.) for i in range(XDim)]) + XDim*log(2) + ld
    return array(invc)

def eDiffMu(X,XDim,NK,betak,m,W,xd,vk,N,K):
    Mu = zeros((N,K))
    for n in range(N):
        for k in range(K):
            A = XDim / betak[k] #shape: (k,)
            B0 = reshape((X[n,:] - m[k,:]),(XDim,1))
            B1 = dot(W[k], B0)
            l = dot(B0.T, B1)
            assert shape(l)==(1,1),shape(l)
            Mu[n,k] = A + vk[k]*l #shape: (n,k)
    
    return Mu



def ePi(tau_pi0,tau_pi1,K):
    exp_ln_pi = zeros(K)
    acc = digamma(tau_pi0) - digamma(tau_pi0 + tau_pi1)
    for k in range(K): exp_ln_pi[k] = digamma(tau_pi1[k]) - digamma(tau_pi0[k] + tau_pi1[k]) + acc[:k].sum()
    return exp_ln_pi        

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
    return exp_z
    
def eFowardsZ(exp_ln_pi,exp_ln_a,XDim,exp_invc,exp_diff_mu,vk,betak,N,K):
    ln_alpha_exp_z = zeros((N,K)) - inf
    #initial state distribution:
    #print 'exp_invc',exp_invc
    ln_alpha_exp_z[0,:] = exp_ln_pi + 0.5*exp_invc - 0.5*vk*exp_diff_mu[0,:] - XDim/(2.*betak)  
    for n in range(1,N):
        for i in range(K): #marginalise over all possible previous states:
            ln_alpha_exp_z[n,:] = logaddexp(ln_alpha_exp_z[n,:], ln_alpha_exp_z[n-1,i]+ exp_ln_a[i,:] + \
                                            0.5*exp_invc - XDim/(2.*betak) - 0.5*vk*exp_diff_mu[n,:])
#        for j in range(K):
#            for i in range(K):
#                ln_alpha_exp_z[n,j] = logaddepx()
    return ln_alpha_exp_z 
    
def eBackwardsZ(exp_ln_pi,exp_ln_a,XDim,exp_invc,exp_diff_mu,vk,betak,N,K):
    ln_beta_exp_z = zeros((N,K)) - inf
    #final state distribution:
    ln_beta_exp_z[N-1,:] = zeros(K)
    for n in range(N-2,-1,-1):
        for j in range(K): #marginalise over all possible next states:
            ln_beta_exp_z[n,:] = logaddexp(ln_beta_exp_z[n,:], ln_beta_exp_z[n+1,j] + exp_ln_a[:,j] + \
                                           0.5*exp_invc - XDim/(2.*betak) - 0.5*vk*exp_diff_mu[n+1,j]) #n+1,j??
    return ln_beta_exp_z

def eS(exp_ln_a, ln_alpha_exp_z, ln_beta_exp_z, (XDim,exp_invc,exp_diff_mu,vk,betak), N, K):
    alpha_exp_z = lnNorm(ln_alpha_exp_z,axis=1)
    beta_exp_z = lnNorm(ln_beta_exp_z,axis=1)
    
    ln_exp_s = zeros((N-1,K,K)) #we only care about the transitions, which is why the length of this var is (N-1)
    for n in range(N-1):
        n1 = n+1 #the 'n' we refer to when using alpha,beta,X
        #ln_exp_s[n,:,:] += exp_ln_a
        for i in range(K):
            for j in range(K):
                #print 'prev_z_%i,next_z_%i,lik_%i_%i'%(i,j,n,j),ln_alpha_exp_z[n1-1,i],ln_beta_exp_z[n1,j],0.5*exp_invc[j] - XDim/(2.*betak[j]) - 0.5*vk[j]*exp_diff_mu[n1,j]
                ln_exp_s[n,i,j] = exp_ln_a[i,j] + ln_alpha_exp_z[n1-1,i] + ln_beta_exp_z[n1,j] + 0.5*exp_invc[j] - XDim/(2.*betak[j]) - 0.5*vk[j]*exp_diff_mu[n1,j]
                #exp_ln_a[i,j] + 
                #ln_exp_s[n,i,j] = ln_alpha_exp_z[n1-1,i] + ln_beta_exp_z[n1,j] # + 0.5*exp_invc[j] - XDim/(2.*betak[j]) - 0.5*vk[j]*exp_diff_mu[n1,j]
    #normalise and exponentiate the rows:
    exp_s = lnNorm(ln_exp_s, axis=2) #shape (N-1,K,K)
    print 'ln_exp_s[10,:,:]',ln_exp_s[10,:,:]
    print 'exp_s[10,:,:]',exp_s[10,:,:]
    exp_s10 = ln_exp_s[10,:,:].copy()
    exp_s10 -= reshape(exp_s10.max(axis=1),(K,1))
    exp_s10 = exp(exp_s10)/reshape(exp(exp_s10).sum(axis=1),(K,1))
    print 'normal exp_s1[10,:,:]',exp_s10
    return exp_s

if __name__ == "__main__":
    from testing.run import test1
    test1()