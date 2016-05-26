
# coding: utf-8
import numpy as np
from sklearn import preprocessing

X = np.arange (1,13,dtype = float)
X = X.reshape(3,4)
A_ = preprocessing.normalize(X, norm='l2',axis=0)

print X

print A_

G = np.dot(A_.T, A_)

print G

# ------------------------------------- Chapter 6 sparse representation --------------------------
# --------------------- using OMP (Orthogonal Matching Pursuit) to seek spark of the A --------------------------
# -------
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
n_samples, n_features = 50, 100
np.random.seed(0)
# Ax = b
# : int, optional
# Desired number of non-zero entries in the solution. If None (by default) this value is set to 10% of n_features.

# A = np.arange (1,n_samples*n_features+1,dtype = float)
# A = A.reshape(n_samples,n_features)
A = np.random.rand(n_samples,n_features)
x = np.zeros(n_features)

print "Random matrix A is shape(50,100)"
print A
# the constraint condition x_i = 1
for i in xrange(n_features):
    x[i] = 1.0
# x = np.arange(1,n_features+1,dtype = float)

G = np.dot(A_.T, A_)
# b = np.zeros((n_features)) 
b = np.dot(A, x) + np.random.rand(n_samples) * 0.1

# A^X^ = b^ 
b_head = np.zeros((n_samples,))
A_head = np.eye(n_samples,n_features) 

    
omp = OrthogonalMatchingPursuitCV()
# omp.fit (A_head, b_head)
omp.fit (A, b)
coef = omp.coef_
# coef = omp.n_nonzeros_coefs
# idx_r, = coef.nonzero()
# print coef.shape
idx_r, = coef.nonzero()
print "the index of the nonzero x_i"
print idx_r
print "the nonzeros solution of the AX = b"
print coef[idx_r]


print "\n"

print "all the solutions" 
print coef
print "\n"

from numpy import linalg as LA
from scipy.optimize import minimize 


x0 = coef[idx_r]

# print "spark(A) = "
# spark_A = LA.norm(x0, ord = 0)
# print spark_A

def rosen(x):
    i, = x.nonzero()
    return LA.norm(x[i],ord=0)

res = minimize (rosen, x0)

print "the status of minization"
print res
spark_A = res.fun
print "spark(A) = "
print spark_A



