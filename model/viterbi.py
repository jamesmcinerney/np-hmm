'''
Created on Oct 13, 2013

@author: James McInerney
'''

from numpy import *

seterr(divide='ignore')

def viterbiLog(ln_lik_obs,exp_a,exp_pi,VERBOSE=False):
    #returns the most likely sequence of hidden states (given the observations) using the Viterbi algorithm

    (N,K) = shape(ln_lik_obs)
    
    #initialize T1 and T2
    T1,T2 = zeros((N,K)), zeros((N,K))
    T1[0,:] = log(exp_pi) + ln_lik_obs[0,:]
    
    #go through data and update T1 and T2
    for n in range(1,N):
        for s in range(K):
            #v = p(z_n-1 = k) * p(z_n = s | z_n-1 = k) * p(x_n | z_n = s)
            #  = p(x_n, z_n = s, z_n-1 = k)
            #  (where s and x_n are given)
            v = T1[n-1,:] + log(exp_a[:,s]) + ln_lik_obs[n,s]
            #if VERBOSE: print 'v_ns,lik',v,lik(s,xs[n],O)
            T1[n,s] = v.max() #find max_k p(x_n ,z_n=s, z_n-1=k)
            T2[n,s] = v.argmax() #find argmax_k p(x_n, z_n=s, z_n-1=k)
        #normalise:
        T1[n,:] -= T1[n,:].max()
        T1[n,:] = log(exp(T1[n,:])/exp(T1[n,:]).sum())

    #find the most likely final state:
    S = zeros(N)
    S[N-1] = T1[N-1,:].argmax()
    for n in range(N-1,-1,-1):
        S[n-1] = T2[n,S[n]]
    return S