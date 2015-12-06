# -*- coding: utf-8 -*-
"""
HW4
Stephen Fang
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi
sns.set_style('white')
sns.set_context('paper')

import os, struct
from array import array
from cvxopt.base import matrix

def read(digits, dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array("B", fimg.read())
    fimg.close()

    ind = [ k for k in xrange(size) if lbl[k] in digits ]
    images =  matrix(0, (len(ind), rows*cols))
    labels = matrix(0, (len(ind), 1))
    for i in xrange(len(ind)):
        images[i, :] = img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]
        labels[i] = lbl[ind[i]]

    return images, labels

"""
# 2 
# plot chi distributions
plt.figure(figsize=(7,4))
x_min, x_max = 0, 4
y_min, y_max = 0, 1
plt.clf()
xx = np.linspace(x_min, x_max, 1000)
for df in [1,2,3,4,5]:
    yy = chi.pdf(xx, df=df)
    plt.plot(xx,yy, label='df = '+str(df)) 
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title('Chi Probability Density Functions',fontsize=11)
plt.xlabel(u'$x$',fontsize=10)
plt.ylabel(u'$p(x)$',fontsize=10)
plt.legend(loc='upper right', fontsize=10)
plt.savefig('Chi Distributions.png', dpi=200)

# simulate many multivariate Gaussians and create histograms
plt.figure(figsize=(7,6))
x_min, x_max = 0, 7
#y_min, y_max = 0, 1
Nmc = 100000
plt.clf()    
xx = np.linspace(x_min, x_max, 1000)
count = 0
for d in [2,5,10,15]:
    count += 1
    plt.subplot(2,2,count)
    yy = chi.pdf(xx, df=d)
    plt.plot(xx, yy, 'r', label='pdf, df = '+str(d))
    mean = [0]*d
    cov =  np.diag([1]*d)
    samples = np.random.multivariate_normal(mean, cov, size=Nmc)
    # calculate distances
    dist = np.sqrt((samples**2).sum(axis=1))
    plt.hist(dist, bins=50, color='b', normed=True, label='Histogram')
    plt.title('Dimension = '+str(d),fontsize=11)    
    plt.xlabel(u'$x=$ Distance from Origin',fontsize=9)
    plt.ylabel(u'$p(x)$',fontsize=9)
    plt.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig('Distances and Chi Distributions.png', dpi=200)
"""

# 3
images, labels = read(range(10), 'training')
images = np.array(images)
#print images.shape # 60,000 x 784
labels = np.array(labels)
#print labels.shape # 60,000 x 1

# divide images by 255 to normalize
images = images/255.

def kmeans(X, k, centers='random'):    
    n, d = X.shape 
    if centers is 'random':
        # randomly initialize the responsibilities
        r = np.random.randint(k, size=n)
        # find mean of each cluster
        centers = np.empty((k,d))
        for i in range(k):
            centers[i] = np.mean(X[r==i,:], axis=0)
    count = 0
    obj = []
    r = np.zeros(n)
    while True:
        old_centers = centers.copy()
        old_r = r.copy()
        error = 0
        for i in range(n):
            x = X[i]
            # calculate distances from centers
            dist = [[c[0], np.linalg.norm(x-c[1])] for c in enumerate(old_centers)]
            # find nearest center
            mindist = min(dist, key=lambda x: x[1]) 
            r[i] = mindist[0] # update responsibilities
            error += mindist[1]**2 # add current squared distance to error
        obj.append(error) # update obj fcn value
        # find new mean of each cluster
        centers = np.empty((k,d))
        for i in range(k):
            centers[i] = np.mean(X[r==i,:], axis=0)
        count += 1
        if (count%10==0):
            print 'Iteration', count    
#        if np.allclose(old_centers, centers):
#            break
        if (old_r==r).all():
            break

    return centers, r, obj, count

def kmeanspp(X, k):
    n, d = X.shape 
    # initialize centers
    centers = np.empty((k,d))
    # initialize distances to closest centers
    dn = np.empty(n)
    # choose an int at random and make it the first center
    i = np.random.randint(n)
    centers[0] = X[i]
    for i in range(1,k):
        for j in range(n):
            x = X[j]
            # calculate distances from centers
            # save distance to the closest center
            dn[j] = min([np.linalg.norm(x-c) for c in centers[:i]])

        # compute distribution proportional to dn**2
        pn = dn**2/np.sum(dn**2)
        # draw a datum from this distribution to be the next center
        centers[i] = X[np.random.multinomial(1,pn)==1]
    return kmeans(X, k, centers) 

def printImage(array):
    assert len(array) == 28**2
    x = array.reshape((28,28))
    plt.imshow(x)
    plt.axis('off')
    plt.margins(0)

#for k in [4,9,16]:
for k in [16]:
    print 'k =', k
    centers, r, obj, count = kmeans(images, k)
    print 'K-Means'
    print 'total iter =', count
    centers_pp, r_pp, obj_pp, count_pp = kmeanspp(images, k)
    print 'K-Means++'
    print 'total iter =', count_pp
    
    # plot the k-means obj fcn as a function of iterations
    plt.figure(figsize=(7,4))
    plt.plot(np.log(obj),'b',label='K-Means')
    plt.plot(np.log(obj_pp),'r',label='K-Means++')    
    plt.legend(fontsize=10)
    plt.title('K-Means Log Objective Function with k = '+str(k), fontsize=11)
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Log Objective', fontsize=10)
    plt.savefig('Kmeans Obj Fcn Values k= '+str(k)+' .png', dpi=200)
    
    # plot cluster centers
    s = int(k**.5)
    plt.figure(figsize=(s,s))
    plt.clf()
    for i in range(k):    
        plt.subplot(s,s,i+1)
        printImage(centers[i])
    plt.tight_layout()
    plt.savefig('Centers k = '+str(k)+'.png', dpi=200)
    
    # print representative images
    for i in range(k):
        s = int(k**.5)
        plt.figure(figsize=(3,3))
        plt.clf()
        for j in range(16):
            plt.subplot(4,4,j+1)
            printImage(images[r==i][j]) 
        plt.tight_layout()
        plt.savefig('Samples in cluster '+str(i+1)+' for k = '+str(k)+'.png', dpi=200)

      
"""
# 4 
# run kmeanspp
k = 4
centers, r, obj, count = kmeanspp(images[:1000,:], k)
print 'k =', k
print 'total iter =', count 
"""


    
