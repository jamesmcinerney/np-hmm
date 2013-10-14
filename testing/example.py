
from model import sensors, general_inf
from testing import viz, synth
from numpy import shape

N, K_grnd, K = 1000, 3, 6

X,Y,mu_grnd = synth.genHMM(N,K_grnd,mu_sd_factor=0.1)

(N,XDim) = shape(X)

#----------------
gSensor = sensors.MVGaussianSensor(K,XDim)
dSensor = sensors.DiscreteSensor(K)
exp_z,_,exp_a,Zmax = general_inf.infer(N,[X,Y],K,[gSensor,dSensor],useMix=1,useHMM=1,plotSensor=gSensor,plotX=X,mu_grnd=mu_grnd)
#----------------

exp_mu, exp_C = gSensor._m, gSensor.expC()

viz.dynamicObs(X,Zmax,exp_z,exp_mu=exp_mu,exp_C=exp_C,waitTime=0.02)
