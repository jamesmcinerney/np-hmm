'''
Created on 3 Oct 2013

@author: James McInerney
'''

from numpy import *

#generate synthetic data

def genHMM(N,K,XDim=2,mu_sd_factor=1,L=10):
    #N: number of observations
    #K: number of discrete components
    #XDim: dimensionality of observed data (default = 2)
    
    #observations/latent variables: loc, day of week, departure from routine:
    X = zeros((N,XDim))
    Y = zeros((N,L))
    Z = zeros((N,K))
    #hyperparameters:
    alpha = 0.05*ones((K,K)) + eye(K) #'sticky' states
    beta = 0.1 #intial dist params.
    mu0 = zeros(XDim) #mean prior for mean
    muC0 = eye(XDim) #covariance matrix prior for mean
    C0a, C0b = 1., 0.9
    
    #generative model:
    #step 1. draw parameters from priors
    pik = random.dirichlet(beta*ones(K)) #distribution over initial state p(l_0)
    #A = array([random.dirichlet(alpha[i,:]) for i in range(K)]) #transition matrix p(z_n | z_n-1)
    A = alpha / reshape(alpha.sum(axis=1),(K,1))
    #mu = array([random.multivariate_normal(mu0,muC0) for _ in range(K)]) #mean parameters for observations
    #C = array([[1/random.gamma(C0a,C0b,XDim) for _ in range(XDim)] for _ in range(K)]) #covariance parameters for observations
    mu = array([random.multivariate_normal(zeros(XDim),mu_sd_factor*eye(XDim)) for _ in range(K)]) #10*
    C = array([0.1*eye(XDim) for _ in range(K)])
    pr_y = array([random.dirichlet(0.1*ones(L)) for _ in range(K)])
    
    #draw observations:
    for n in range(N):
        #draw latent component:
        if n==0:
            Z[n,:] = random.multinomial(1,pik)
        else:
            prev_z = Z[n-1,:].argmax()
            Z[n,:] = random.multinomial(1,A[prev_z,:])
        z_n = Z[n,:].argmax()
        #draw observation for time step n:
        X[n,:] = random.multivariate_normal(mu[z_n,:], C[z_n,:,:])
        Y[n,:] = random.multinomial(1,pr_y[z_n,:])
        
    return X,Y,mu

def genHMM1(N,K,XDim=2,mu_sd_factor=1):
    #N: number of observations
    #K: number of discrete components
    #XDim: dimensionality of observed data (default = 2)
    
    #observations/latent variables: loc, day of week, departure from routine:
    X = zeros((N,XDim))
    Z = zeros((N,K))
    #hyperparameters:
    alpha = 0.05*ones((K,K)) + eye(K) #'sticky' states
    beta = 0.1 #intial dist params.
    mu0 = zeros(XDim) #mean prior for mean
    muC0 = eye(XDim) #covariance matrix prior for mean
    C0a, C0b = 1., 0.9
    
    #generative model:
    #step 1. draw parameters from priors
    pik = random.dirichlet(beta*ones(K)) #distribution over initial state p(l_0)
    #A = array([random.dirichlet(alpha[i,:]) for i in range(K)]) #transition matrix p(z_n | z_n-1)
    A = alpha / reshape(alpha.sum(axis=1),(K,1))
    #mu = array([random.multivariate_normal(mu0,muC0) for _ in range(K)]) #mean parameters for observations
    #C = array([[1/random.gamma(C0a,C0b,XDim) for _ in range(XDim)] for _ in range(K)]) #covariance parameters for observations
    mu = array([random.multivariate_normal(zeros(XDim),mu_sd_factor*eye(XDim)) for _ in range(K)]) #10*
    C = array([0.1*eye(XDim) for _ in range(K)])
    
    #draw observations:
    for n in range(N):
        #draw latent component:
        if n==0:
            Z[n,:] = random.multinomial(1,pik)
        else:
            prev_z = Z[n-1,:].argmax()
            Z[n,:] = random.multinomial(1,A[prev_z,:])
        z_n = Z[n,:].argmax()
        #draw observation for time step n:
        X[n,:] = random.multivariate_normal(mu[z_n,:], C[z_n,:,:])
        
    return X,Z,A,mu