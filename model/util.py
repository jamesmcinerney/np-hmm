'''
Created on 24 Sep 2013

@author: James McInerney
'''

from numpy import *
from numpy.linalg.linalg import inv
from scipy.special import gamma

def gam(xs,k,theta):
    #gamma distribution from wikipedia defn:
    N = len(xs)
    ln_ys = zeros(N) - log(gamma(k)) - k*log(theta)
    for n in range(N): ln_ys[n] += (k-1.)*log(xs[n]) - (xs[n])/theta
    return exp(ln_ys)



def getTailNum(s):
    N = len(s)
    i=0
    while i<N and not(is_num(s[i])): i+=1
    if i<N: return s[i:]
    else: return ''
        

def is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def listToArr(xs):
    seen = []
    for x in xs:
        if x not in seen: seen.append(x)
    L = len(seen)
    N = len(xs)
    X = zeros((N,L))
    for n in range(N):
        l = seen.index(xs[n])
        X[n,l] = 1
    return X

def lnNorm(lnX_,axis=1):
    #normalise log probabilities by axis
    lnX = lnX_.copy()
    dims=shape(lnX)
    dimsFlat = list(dims)
    dimsFlat[axis] = 1
    lnX -= reshape(lnX.max(axis=axis),dimsFlat)
    X = exp(lnX) / reshape(exp(lnX).sum(axis=axis),dimsFlat)
    return X

def testLnNorm():
    X = zeros((2,3,3))
    X[0,0,0]=10
    X[0,1,0]=3
    X[0,0,1]=1
    X[0,1,1]=4
    X[0,2,1]=1
    X[1,0,0]=5
    X[1,1,0]=5
    X[1,0,1]=3
    X[1,1,1]=1
    X[1,2,2]=1
    print 'X[0,:,:]\n',X[0,:,:]
    print 'X[1,:,:]\n',X[1,:,:]
    lnX = log(X)
    print 'lnNorm X[0,:,:]\n',lnNorm(lnX,axis=2)[0,:,:]
    print 'lnNorm X[1,:,:]\n',lnNorm(lnX,axis=2)[1,:,:]
    
def log0(x):
    #returns -inf if x==0, otherwise log_e(x)
    if x==0: return -1e6
    else: return log(x)
    
def doIf(f,cond,a,b):
    if cond: return f(a,b)
    else: return a
    
def slottedArr(Z,T):
    N = len(T) #number of obs
    (NS,K) = shape(Z) #number of time slots x number of components
    Z1 = zeros((N,K))
    t = 0 #position in X, T and ln_obs_uns
    for n in range(NS): #for each time slot
        while t<N and T[t]==n:
            Z1[t,:] = Z[n,:] #product of likelihoods for same time slot
            t+=1 
    print 'shape(Z1)',shape(Z1)
    print 'shape(Z)',shape(Z)
    print '(N,K)',(N,K)
    
    assert all((Z1.sum(axis=1)-1)**2<1e-5),Z1.sum(axis=1)
    return Z1

def inv0(X):
    #inverts X if it is a matrix, otherwise, it finds numerical inverse
    try:
        Xm1 = inv(X)
        return Xm1
    except IndexError:
        #print 'reverting to 1D for inverse matrix'
        return 1/float(X)
    
if __name__ == "__main__":
    testLnNorm()