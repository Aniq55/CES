import numpy as np
import pickle


# Parameters
n = 10
p_0 = 0.2
theta_0 = 3e-5
mu_0 = 5.5
sigma_0 = 1
sigma_1 = 1.5
sigma_2 = 0.7


# step 1
A = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        A[i][j] = np.random.binomial(1, p_0)

# step 2
for i in range(n):
    A[i][i] = 1


# step 4
Lambda = np.zeros((n,1))
for i in range(n):
    Lambda[i] = np.random.lognormal(mu_0, sigma_0**2)


# step 5
Eta = np.zeros((n,1))
for i in range(n):
    Eta[i] = np.random.normal(0, sigma_1**2)


# steps 6-10
Alpha = np.zeros((n,n))
Beta = np.zeros((n,n))
Gamma = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        if A[i][j] == 1:
            Beta[i][j] = np.random.exponential(theta_0)
            Alpha[i][j] = Beta[i][j]/Lambda[j]
            Gamma[i][j] = np.random.normal(0, sigma_2**2)


# step 11
Theta = {
    'n': n,
    'Lambda': Lambda,
    'Eta': Eta,
    'Alpha': Alpha,
    'Beta': Beta,
    'Gamma': Gamma
}

with open('/home/chri6578/Documents/CES/params/Theta_2.pickle', 'wb') as file:
    pickle.dump(Theta, file)


