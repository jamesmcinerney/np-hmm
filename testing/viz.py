'''
Created on 3 Oct 2013

@author: James McInerney
'''

#visualising HMM (and inference/prediction)


from matplotlib.pyplot import *
from numpy import *
import time
from testing.util import create_cov_ellipse
from model.util import listToArr
try:
    from mpl_toolkits.basemap import Basemap
except ImportError:
    print 'no Basemap'
import sys
from testing.googlemaps import snippet

#where to save animation images:
ANIM_ROOT = '/media/8A8823BF8823A921/vb-ihmm-anim/'

def createGoogleMap(X,ZS,exp_mu=None,exp_C=None):
    fpath = 'googlemaps/webdocs/mapPlot.html'
    #find active components:
    #visitThreshold = 0.5
    #(ks,)=where(exp_z.sum(axis=0)>visitThreshold)
    (ks,) = where(ZS.sum(axis=0)>0) 
    if exp_mu is not None and exp_C is not None:
        #create ellipses from the parameters of the Gaussians:
        ellipses = createEllipses(exp_mu,exp_C,ks)
    else:
        ellipses = None
    (x_min,y_min),(x_max,y_max) = findLim(X,graphMargin=0.)
    #zs indicates current component at any time (of only the active ellipses)
    (N,K) = shape(ZS)
    zs = []
    for n in range(N): zs.append(list(ks).index(ZS[n,:].argmax()))
    snippet.write(X,zs,ZS,exp_mu,y_min,y_max,x_min,x_max,fpath,kInit=ks[0],ellipses=ellipses)
    print 'written google maps to file',fpath

def createEllipses(exp_mu,exp_C,ks):
    #takes set of params and makes a list of ellipse parameters:
    ellipses = [] #list of ellipse parameters
    f = 200 #500 #constant scaling factor to make ellipses visible (todo: make 95% confidence range)
    for k in ks: 
        e= create_cov_ellipse(exp_C[k], exp_mu[k,:],color='r',alpha=0.3)
        #ellipse example: (lat,lng,width,height,rotation)
        ellipses.append((e.center[1],e.center[0],f*e.height,f*e.width,-e.angle))
    return ellipses

    
def dynamicObs(X,Zmax,exp_z,mu_grnd=None,exp_mu=None,exp_C=None,waitTime=0.02,pathLen=4):
    ion()    
    fig = figure(figsize=(10,10))
    ax_spatial = fig.add_subplot(1,1,1) #http://stackoverflow.com/questions/3584805/in-matplotlib-what-does-111-means-in-fig-add-subplot111
    circs = []

    (N,XDim) = shape(X)
    assert XDim==2
    (N1,K)  = shape(exp_z)
    assert N1==N
    NK = exp_z.sum(axis=0)
    alpha0 = 1.0
    
    markers = ['x','o','^','v','d'] #,'d'] #markers for different inferred component assignments
    sct = [] #different scatter style for each data point
    (ks,) = where(exp_z.sum(axis=0)>=0.5) #find the components that actually appear in the data (not need to visualise the rest)
    ks = list(ks)
    for k in range(len(ks)): sct.append(scatter(9999.,9999.,marker=markers[k%len(markers)]))
    if mu_grnd is not None:
        (K_grnd,_) = shape(mu_grnd)
        for k in range(K_grnd): scatter(mu_grnd[k,0],mu_grnd[k,1],color='w',marker='d',s=50) #plot ground truth means
    #partition the data:
    Xi,ni = [], []
    for k in ks:
        (ns,)=where(exp_z.argmax(axis=1)==k)
        Xi.append(X[ns,:])
        ni.append(0) #current position in each partition
    if exp_mu is not None:
        #show only means corresponding to actual probability mass
        sctZ = scatter(exp_mu[ks,0],exp_mu[ks,1],color='r') #plot the inferred means

    (x_min,y_min),(x_max,y_max) = findLim(X)
    xlim(x_min,x_max)
    ylim(y_min,y_max)
    print 'X min max',(X[:,0].min(),X[:,1].min()),(X[:,0].max(),X[:,1].max())
    path1 = None #the stored path of recent points
    
    for n in range(N):
        #work out which component current data point belongs to:
        i = Zmax[n]
        ik = ks.index(i) #look up scatter plot that we need to update
        print 'x_n, latent loc',X[n,:],i
        Xi_ = Xi[ik]
        n_ = ni[ik]+1
        sct[ik].set_offsets(Xi_[:n_,:])
        ni[ik] = n_
        nprev = max(0,n-pathLen)
        if path1 is not None: path1.remove()
        path1, = plot(X[nprev:n+1,0],X[nprev:n+1,1],color='b')
        if exp_mu is not None:
            #ellipses to show covariance of components
            for circ in circs: circ.remove()
            circs = []
            for k in ks:
                circ = create_cov_ellipse(exp_C[k], exp_mu[k,:],color='r',alpha=0.3) #calculate params of ellipses (adapted from http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals)
                circs.append(circ)
                #add to axes:
                ax_spatial.add_artist(circ)

        try:
            savefig(ANIM_ROOT + 'animation/%04d.png'%n)
        except IOError:
            0 #print 'could not save file, IOError'
        time.sleep(waitTime)
        draw()

def findLim(X,graphMargin=0.1):
    #|graphMargin| specifies how much more of the graph we display beyond the corner points
    x_min,x_max = X[:,0].min(), X[:,0].max()
    y_min,y_max = X[:,1].min(), X[:,1].max()
    return (x_min*(1-sign(x_min)*graphMargin), y_min*(1-sign(y_min)*graphMargin)), \
                (x_max*(1+sign(x_max)*graphMargin), y_max*(1+sign(y_max)*graphMargin))


def testMap(X):
    # setup Lambert Conformal basemap.
    # set resolution=None to skip processing of boundary datasets.
    (x_min,y_min),(x_max,y_max) = findLim(X)
    kx,ky = 0.,0.
    (x_min,y_min),(x_max,y_max) = (-1.42352-kx,50.930738-ky),(-0.956601+kx,51.086704+ky)
    x_mid,y_mid = (x_max-x_min)/2., (y_max-y_min)/2.
    m = Basemap(llcrnrlon=x_min,llcrnrlat=y_min,urcrnrlon=x_max,urcrnrlat=y_max,
                resolution='i',projection='tmerc',lon_0=x_mid,lat_0=y_min) #adapted from: http://matplotlib.org/basemap/users/tmerc.html
    m.drawcoastlines()
    #m.bluemarble()
    #m.shadedrelief()
    m.etopo()
    show()
    
if __name__ == "__main__":
    from run import testReal2
    testReal2()
    #test1()
    #testMap()