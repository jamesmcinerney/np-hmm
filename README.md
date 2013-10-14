np-hmm
======

General Bayesian non-parametric hidden Markov model for inference on dynamic data with (unknown) number of hidden states.

Specify which sensors you want to use (any combination of multivariate Gaussian or discrete) on data.

4 lines to define and run inference on a custom multi-modal non-parametric hidden Markov model:

```python
K = 10 #state truncation parameter
gSensor = sensors.MVGaussianSensor(K,2) #to apply to 2 dimensional continuous data X1
dSensor = sensors.DiscreteSensor(K) #to apply to discrete data X2
exp_z,_,exp_a,Zmax = general_inf.infer(N,[X1,X2],K,[gSensor,dSensor])
```

Returns hidden state assignments (exp_z), transition matrix (exp_a), and most likely state path using the Viterbi algorithm (Zmax).

To see full example, run:

```html
testing/example.py
``` 