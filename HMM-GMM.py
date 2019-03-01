
# coding: utf-8

# In[ ]:


#!/usr/bin/python3
# coding: utf-8

import numpy as np
import math
import sys

# determinant of a 2*2 matrix
def det(M):
    return (M[0,0]*M[1,1] - M[1,0]*M[0,1])

# inverse of a 2*2 matrix
def inv(M):
    return np.array([[M[1,1]/det(M), -1*M[0,1]/det(M)],[-1*M[1,0]/det(M), M[0,0]/det(M)]], dtype=np.float128)

def multivariate_normal(mean, cov, array):
    z = (-1/2)*((array - mean).dot(inv(cov))).dot((array - mean).T)
    n_factor = 1/math.sqrt(det(cov)*(2*math.pi)**cov.shape[0])
    p = n_factor*np.exp(z)
    return p

# compute alpha and beta
def ab(Mu, Sigma, data_array, Pi, num_center, A):
    dim = data_array.shape
    alpha = np.zeros((dim[0], num_center), dtype=np.longdouble)
    for i in range(dim[0]):
        if i == 0:
            alpha[i,:] = np.array([multivariate_normal(Mu[k], Sigma[k], data_array[i,])*Pi[k] for k in range(num_center)], dtype=np.longdouble)      
        else:
            alpha[i,:] = np.array([alpha[i-1,].dot(A[:,k])*multivariate_normal(Mu[k], Sigma[k], data_array[i,]) for k in range(num_center)], dtype=np.longdouble)

    beta = np.zeros((dim[0], num_center), dtype=np.longdouble)
    for i in range(dim[0]-1, -1, -1):
        if i == dim[0]-1:
            beta[i,:] = np.array([1] * num_center, dtype=np.longdouble)
        else:
            b_llh = [multivariate_normal(Mu[k], Sigma[k], data_array[i+1,]) for k in range(num_center)]*beta[i+1,]
            beta[i,:] = np.array([b_llh.dot(A[k,:]) for k in range(num_center)],dtype=np.longdouble)
    return (alpha, beta)

# new transition matrix
def Transition(num_center, Mu, Sigma, data_array, dim, alpha, beta, A):
    A_n = np.zeros((num_center, num_center), dtype=np.longdouble)
    for n in range(dim[0]):
        for i in range(num_center):
            for j in range(num_center):
                if n != dim[0]-1:
                    A_n[i, j] += alpha[n, i]*beta[n+1, j]*A[i, j]*multivariate_normal(Mu[j], Sigma[j], data_array[n+1])/np.sum(alpha[dim[0]-1,:])
                else:
                    A_n[i, j] += alpha[n, i]*A[i, j]/np.sum(alpha[dim[0]-1,:])
    A_n = A_n/np.sum(A_n, axis=0)
    return A_n

def EM_HMM(num_center, train_data, dev_data, Cov_type = None):
    s_lllh = []
    s_dev_lllh = []

    data_array = np.array(train_data, dtype=np.longdouble)
    dim = data_array.shape
    
    # initializing Mu, Sigma and Pi
    x1 = [min(data_array[:,0]), max(data_array[:,0])]
    x2 = [min(data_array[:,1]), max(data_array[:,1])]
    Mu = []
    for i in range(num_center):
        x1_noise = np.random.uniform(x1[0], x1[1])
        x2_noise = np.random.uniform(x2[0], x2[1])
        Mu.append(np.mean(data_array, axis=0)+[x1_noise, x2_noise])
    Sigma = [np.cov(data_array.T)] * num_center
    Pi = [1/num_center] * num_center
    
    # initialize random transition matrix
    A = np.random.rand(num_center, num_center)
    # normalizing the summation of each row to 1
    A = np.array([A[i,:]/sum(A[i,:]) for i in range(A.shape[0])], dtype=np.longdouble)
    alpha, beta = ab(Mu, Sigma, data_array, Pi, num_center, A) 
    
    # EM training:
    for count in range(40):
        # E-M algorithm E step
        # rnk matrix
        rnk = np.multiply(alpha, beta)/np.sum(alpha[dim[0]-1,:])
        
        # E-M algorithm M step
        Mu = np.array([data_array.T.dot(rnk[:,k])/sum(rnk[:,k]) for k in range(num_center)], dtype=np.longdouble)
        Sigma = [np.zeros((2, 2))] * num_center
        for i in range(dim[0]):
            for k in range(num_center):
                Sigma[k] = Sigma[k] + rnk[i, k]*np.outer((data_array[i]-Mu[k]), (data_array[i]-Mu[k]))
        Sigma = [Sigma[i]/sum(rnk[:,i]) for i in range(num_center)]
        # Tied Cov matrices
        if Cov_type == 'Tied':
            Sigma = [sum(Sigma)/num_center for i in range(num_center)]
        Pi = rnk[0,:]
        
        A = Transition(num_center, Mu, Sigma, data_array, dim, alpha, beta, A)
        alpha, beta = ab(Mu, Sigma, data_array, Pi, num_center, A)
        # Log likelihood
        s_lllh.append(np.log(sum((alpha[dim[0]-1,]))))        
             
        # EM testing:
        # Log likelihood of Development data
        dev_array = np.array(dev_data, dtype=np.longdouble)
        dim2 = dev_array.shape
        alpha_d, beta_d = ab(Mu, Sigma, dev_array, Pi, num_center, A)
        s_dev_lllh.append(np.log(sum(alpha_d[dim2[0]-1,])))

    return (s_lllh, s_dev_lllh)

def main():
    contents = []
    with open('points.dat') as f:
        for line in f:
            contents.append(line.strip())
    o_data = [''.join(i).split() for i in contents]      
    x_data = [[eval(i[0]), eval(i[1])] for i in o_data if i != []]
    train_data = x_data[:int(len(x_data)*0.9)+1]
    dev_data = x_data[-int(len(x_data)*0.1):]
    
    clusters = eval(sys.argv[sys.argv.index('--clusters')+1])
    if '--clusters' in sys.argv and '--Tied' in sys.argv:
        result = EM_HMM(clusters, train_data, dev_data, 'Tied')
    elif '--clusters' in sys.argv and '--Tied' not in sys.argv:
        result = EM_HMM(clusters, train_data, dev_data)
    else:
        print('The arguments you inputted were wrong, please refer to README.')
    
    print('The training data likelihood for all 40 iterations are: \n' + repr(result[0]) + '\n')
    print('The development data likelihood for all 40 iterations are: \n' + repr(result[1]) + '\n')

if __name__ == "__main__":
    main()

