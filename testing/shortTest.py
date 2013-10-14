'''
Created on Oct 11, 2013

@author: root
'''


from matplotlib.pyplot import *
markers = ['x','o','^','v','D','d']
for m in markers:
  print m
  scatter(1,1,marker=m)
  show()