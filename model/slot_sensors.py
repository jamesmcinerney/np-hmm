'''
Created on 8 Oct 2013

@author: James McInerney
'''

#extension of Sensors to deal with slotted data (i.e., multiple observations per time slot)

from model.sensors import MVGaussianSensor, Sensor
from numpy import *



class SlottedSensor(Sensor):
    def __init__(self,K,sensor):
        self._sensor = sensor
        self._K = K
    
    #need to aggregate each time slot:
    def loglik(self,txs):
        (T,X) = txs #split into time slot and data itself
        NS = T.max()+1
        K = self._K
        ln_obs_uns = self._sensor.loglik(X)
        #all that remains is to convert the unslotted version to slotted likelihoods:
        ln_obs_lik = zeros((NS,K))
        #requires: T value of each X is in ascending order:
        t = 0 #position in X, T and ln_obs_uns
        for n in range(NS): #for each time slot
            while t<len(T) and T[t]==n:
                ln_obs_lik[n,:] += ln_obs_uns[t,:] #product of likelihoods for same time slot
                t+=1 
#        print 'ln_obs_uns',ln_obs_uns
#        print 'T',T
#        print 'ln_obs_lik',ln_obs_lik[:300,:]
#        print 'shape(ln_obs_lik)',shape(ln_obs_lik)
        return ln_obs_lik
    
    #TODO: NOT SURE IF IT WORKS!
    def m(self,txs,exp_z):
        #idea: repeat exp_z for obs in same time slot
        (T,X) = txs
        (N,XDim) = shape(X)
        (NS,K) = shape(exp_z)
        Z = zeros((N,K))
        t = 0 #position in X, T and ln_obs_uns
        for n in range(NS): #for each time slot
            while t<len(T) and T[t]==n:
                Z[t,:] = exp_z[n,:] #product of likelihoods for same time slot
                t+=1         
        self._sensor.m(X,Z)
        
if __name__ == "__main__":
    from testing.run import testReal2
    testReal2()
