import numpy as np
import itertools
import logging
import pickle 
import pandas as pd 
from tqdm import tqdm

def create_logger(name, log_file, level):
    # Create a custom logger for each log file
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create a file handler for the logger
    handler = logging.FileHandler(log_file)
    handler.setLevel(level)
    
    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    
    # Add the handler to the logger
    if not logger.handlers:  # Avoid adding multiple handlers in repeated calls
        logger.addHandler(handler)
    
    return logger

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

def total_count(E):
    return np.sum([len(E[e]) for e in E.keys()])

def strip_time(E):
    flattened = [(i, value) for i in E for value in E[i]]
    sorted_by_value = sorted(flattened, key=lambda x: x[1])
    sorted_indices = [i for i, _ in sorted_by_value]
    
    return sorted_indices
    
def order_prob_vector(L, n, k):
    # Generate all ordered sequences of size k with elements from 0 to n-1
    all_sequences = list(itertools.product(range(n), repeat=k))
    
    # Create a dictionary to map each sequence to an index in the count vector
    sequence_to_index = {seq: idx for idx, seq in enumerate(all_sequences)}
    
    # Initialize a count vector of size n^k
    count_vector = [0] * len(all_sequences)
    
    # Slide a window of length k across L and count occurrences of each sequence
    for i in range(len(L) - k + 1):
        seq = tuple(L[i:i + k])
        if seq in sequence_to_index:
            count_vector[sequence_to_index[seq]] += 1

    return count_vector/np.sum(count_vector)    

# Causal Model Generator
def CMG(n, p_0, theta_0, mu_0, sigma_0, sigma_1, sigma_2):
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
    
    return Theta

# Event Sequence Generator
def ESG(Theta):
    n = Theta['n']
    Lambda = Theta['Lambda']
    Eta = Theta['Eta']
    Alpha = Theta['Alpha']
    Beta = Theta['Beta']
    Gamma = Theta['Gamma']
    T = 1e4

    # step 2
    PHI = {}

    for i in range(n):
        PHI[i] = []
        t = 0
        t_prev = 0
        while t_prev < T:
            t = np.random.exponential(Lambda[i])[0]
            PHI[i].append(t + t_prev)
            t_prev = t + t_prev

    # step 3
    tuple_list = [(i, t) for i in range(n) for t in PHI[i]]
    T_list = sorted(tuple_list, key=lambda x: x[1])

    # steps 4-11
    E_all = {}
    for i in range(n):
        E_all[i] = []
    t_last = np.zeros((n,))

    for (e,t) in T_list:
        # update x_i^t
        s = np.sum( Gamma[e]*np.power(t- t_last, Alpha[e]-1)*np.exp(-(t-t_last)*Beta[e])*(t_last != 0) ) + Eta[e]
        p = sigmoid(s)
        # print(s, p)
        trial = np.random.binomial(1,p)
        if trial:
            E_all[e].append(t)
            t_last[e] = t
            
    return E_all


def ES2DF(ES):
    rows = []
    for key, values in ES.items():
        for value in values:
            rows.append({'event': key, 'timestamp': int(value)})

    df = pd.DataFrame(rows)
    df = df.sort_values(by='timestamp', ascending=True)
    df = df.reset_index(drop=True)
    
    return df

# Causal discovery
def estimate(PHI, E_all, tau_bar, n):

    Y0 = np.zeros((n, 2**n))
    Y1 = np.zeros((n, 2**n))

    for i in tqdm(range(n)):
        for t in PHI[i]:
            z = np.zeros((n,))
            for j in range(n):
                z[j] = np.sum((E_all[j] > t - tau_bar)*(E_all[j] < t)) > 0
                
            if t in E_all[i]:
                Y1[i][int(z.T @ pow2)] +=1
            else:
                Y0[i][int(z.T @ pow2)] +=1
                
    Y_sum = Y0 + Y1 

    P1 = Y1 / Y_sum

    # check impact of j on i
    A_est = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            effect = []
            for k in range(2**n):
                if k & (1 << j) == 0:
                    # j is present
                    effect.append(P1[i][k+pow2[j]]- P1[i][k])
            if np.isnan(np.nanmean(effect)) == 0:
                A_est[i][j] = np.nanmean(effect)
                
    return A_est 

# GENERATE PARAMS (CAUSAL MODEL)
def gen_causal_model(n, p_0, mu_0, sigma_0, sigma_1, param_name):
    # Parameters
    # n = 10
    # p_0 = 0.2
    # mu_0 = 2
    # sigma_0 = 1
    # sigma_1 = 1e1


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

    # steps 6-10
    Gamma = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if A[i][j] == 1:
                Gamma[i][j] = np.random.normal(0, sigma_1**2)


    # step 11
    Theta = {
        'n': n,
        'Lambda': Lambda,
        'Gamma': Gamma
    }

    with open(f'/home/chri6578/Documents/CES/params/{param_name}.pickle', 'wb') as file:
        pickle.dump(Theta, file)
        
        
# LOAD PARAMS
def load_params(param_name):
    with open(f'/home/chri6578/Documents/CES/params/{param_name}.pickle', 'rb') as file:
        Theta = pickle.load(file)

    n = Theta['n']
    Lambda = Theta['Lambda']
    Gamma = Theta['Gamma']
    
    return (n, Lambda, Gamma)


# GENERATE SEQUENCE (SIMPLE)
def gen_sequence_simple(n, Lambda, Gamma, T, bar_tau_0):
    
    n = n 
    Lambda = Lambda 
    Gamma = Gamma
    
    # step 2
    PHI = {}

    for i in range(n):
        PHI[i] = []
        t = 0
        t_prev = 0
        while t_prev < T:
            t = np.random.exponential(Lambda[i])[0]
            PHI[i].append(t + t_prev)
            t_prev = t + t_prev

    # step 3
    tuple_list = [(i, t) for i in range(n) for t in PHI[i]]
    T_list = sorted(tuple_list, key=lambda x: x[1])

    # steps 4-11

    E_all = {}
    for i in range(n):
        E_all[i] = []
    t_last = np.zeros((n,))

    for (e,t) in T_list:
        
        s = np.sum( Gamma[e]*(t_last != 0)*(t_last < t)*(t_last > t - bar_tau_0) )
        p = sigmoid(s)
        trial = p >= 0.5
        
        # update x_i^t
        if trial:
            E_all[e].append(t)
            t_last[e] = t
            
    return (PHI, E_all)


# CAUSAL DISCOVERY
def estimate(PHI, E_all, tau_bar, n):
    
    pow2 = np.power(2, np.arange(n))

    Y0 = np.zeros((n, 2**n))
    Y1 = np.zeros((n, 2**n))

    for i in tqdm(range(n)):
        for t in PHI[i]:
            z = np.zeros((n,))
            for j in range(n):
                z[j] = np.sum((E_all[j] > t - tau_bar)*(E_all[j] < t)) > 0
                
            if t in E_all[i]:
                Y1[i][int(z.T @ pow2)] +=1
            else:
                Y0[i][int(z.T @ pow2)] +=1
                
    Y_sum = Y0 + Y1 

    P1 = Y1 / Y_sum

    # check impact of j on i
    A_est = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            effect = []
            for k in range(2**n):
                if k & (1 << j) == 0:
                    # j is present
                    effect.append(P1[i][k+pow2[j]]- P1[i][k])
            if np.isnan(np.nanmean(effect)) == 0:
                A_est[i][j] = np.nanmean(effect)
                
    return A_est 


def search(E_all, tau_bar, n):
    
    T_all = sorted([elem for sublist in E_all.values() for elem in sublist])

    pow2 = np.power(2, np.arange(n))

    Z_exists = np.zeros((2**n,1))

    for t in tqdm(T_all):
        z = np.zeros((n,))
        for j in range(n):
            z[j] = np.sum((E_all[j] > t - tau_bar)*(E_all[j] < t)) > 0
        Z_exists[int(z.T @ pow2)] = 1
        
    return Z_exists


def search_(E_all, tau_bar, n, k):
    binary_str = bin(k)[2:].zfill(n)
    binary_array = np.array([int(bit) for bit in binary_str], dtype=np.int8)
    
    T_all = sorted([elem for sublist in E_all.values() for elem in sublist])
    
    for t in tqdm(T_all):
        z = np.zeros((n,))
        for j in range(n):
            z[j] = np.sum((E_all[j] > t - tau_bar)*(E_all[j] < t)) > 0
        
        if np.linalg.norm(binary_array - z)==0:
            print('done')
            return True 
        
    return False


def estimate_(E_all, tau_bar, n):
    
    pow2 = np.power(2, np.arange(n))
    
    Y1 = {}
    for i in range(n):
        Y1[i] = {}
    

    for i in tqdm(range(n)):
        for t in E_all[i]:
            z = np.zeros((n,))
            for j in range(n):
                z[j] = np.sum((E_all[j] > t - tau_bar)*(E_all[j] < t)) > 0
            
            Y1[i][int(z.T @ pow2)] = 1

    # check impact of j on i
    A_est = np.zeros((n,n))

    
    c = 0
    C = []
    for i in tqdm(range(n)):
        for j in range(n):
            effect = []
            # for k in range(2**n):
            L = list(Y1[i].keys())
            for k in L:
                if k & (1 << j) == 0:
                    
                    c+=2
                    
                    C.append(k+pow2[j])
                    C.append(k)
                        
                    if search_(E_all, tau_bar, n, k+pow2[j]) and search_(E_all, tau_bar, n, k):
                        if k+pow2[j] not in Y1[i]:
                            Y1[i][k+pow2[j]] = 0
                            
                        if k not in Y1[i]:
                            Y1[i][k] = 0
                        
                        effect.append(Y1[i][k+pow2[j]]- Y1[i][k])
                        
            if np.isnan(np.nanmean(effect)) == 0:
                A_est[i][j] = np.nanmean(effect)
                
    K = list(set(C))
    
    
                
    print(c, len(set(C)))
    return A_est 

# Idea: The search_ operation can be performed once and saved as a dictionary (per node)
# even performing it once is quite expensive.

# import numpy as np
# from collections import defaultdict
# from tqdm import tqdm

# def search_(E_all, tau_bar, n, k, T_all_set):
#     """ Optimized search function with set lookup for faster performance. """
#     binary_array = np.unpackbits(np.array([k], dtype=np.uint8).view(np.uint8), bitorder='big')[-n:]
    
#     for t in T_all_set:
#         z = np.zeros(n, dtype=np.int8)
#         for j in range(n):
#             z[j] = np.any((E_all[j] > t - tau_bar) & (E_all[j] < t))
        
#         if np.array_equal(binary_array, z):
#             return True
#     return False


# def estimate_(E_all, tau_bar, n):
#     pow2 = 1 << np.arange(n)  # Faster than 2**np.arange(n)
    
#     Y1 = [defaultdict(int) for _ in range(n)]
    
#     # Precompute T_all as a set for fast lookup
#     T_all_set = {t for sublist in E_all.values() for t in sublist}

#     for i in tqdm(range(n)):
#         for t in E_all[i]:
#             z = np.zeros(n, dtype=np.int8)
#             for j in range(n):
#                 z[j] = np.any((E_all[j] > t - tau_bar) & (E_all[j] < t))
            
#             Y1[i][np.dot(z, pow2)] = 1

#     # Estimate adjacency matrix
#     A_est = np.zeros((n, n))

#     for i in tqdm(range(n)):
#         for j in range(n):
#             effect = []
#             keys_list = list(Y1[i].keys())
            
#             for k in keys_list:
#                 if not (k & (1 << j)):  # If bit j is not set
                    
#                     k_j = k + pow2[j]  # Set bit j
#                     if search_(E_all, tau_bar, n, k_j, T_all_set) and search_(E_all, tau_bar, n, k, T_all_set):
                        
#                         effect.append(Y1[i].get(k_j, 0) - Y1[i].get(k, 0))
            
#             mean_effect = np.nanmean(effect)
#             if not np.isnan(mean_effect):
#                 A_est[i, j] = mean_effect

#     return A_est
