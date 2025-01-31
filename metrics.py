import numpy as np 
from utils import *

# Average Time Difference
def ATD_half(E_1, E_2):
    atd = 0.0
    
    T_1_max = np.max([np.max(E_1[i]) for i in E_1.keys() if len(E_1[i])>0])
    T_2_max = np.max([np.max(E_2[i]) for i in E_2.keys() if len(E_2[i])>0])
    
    T_max = np.max([T_1_max, T_2_max])
    
    for e in E_1.keys():
        sum_e = 0.0
        for t in E_1[e]:
            if len(E_2[e])>0 :
                sum_e += np.min(np.abs(t - E_2[e]))
            else:
                sum_e += T_max
        atd += sum_e / len(E_1)
    
    return atd/len(E_1.keys())


def ATD(E_1, E_2):
    return 0.5*ATD_half(E_1, E_2) + 0.5*ATD_half(E_2, E_1)


# Average Count Difference
def ACD(E_1, E_2, tau):
    acd_1 = 0.0
    acd_2 = 0.0
    
    for e in E_1.keys():
        sum_e_1 = 0.0
        for t in E_1[e]:
            t_left = t - tau 
            t_right = t + tau 
            
            num_e_1 = np.sum((E_1[e] >= t_left) & (E_1[e] <= t_right))
            num_e_2 = np.sum((E_2[e] >= t_left) & (E_2[e] <= t_right))
            
            sum_e_1 += np.abs(num_e_1 - num_e_2)
            
        sum_e_2 = 0.0
        for t in E_2[e]:
            t_left = t - tau 
            t_right = t + tau 
            
            num_e_1 = np.sum((E_1[e] >= t_left) & (E_1[e] <= t_right))
            num_e_2 = np.sum((E_2[e] >= t_left) & (E_2[e] <= t_right))
            
            sum_e_2 += np.abs(num_e_1 - num_e_2)
        
        acd_1 += sum_e_1
        acd_2 += sum_e_2
        
    return 0.5*(acd_1/total_count(E_1) + acd_2/total_count(E_2))


# k-Order Distance
def OD(E_1, E_2, k):
    n = len(E_1.keys())
    
    p_1 = order_prob_vector(strip_time(E_1), n, k)
    p_2 = order_prob_vector(strip_time(E_2), n, k)
    
    return np.sum(np.abs(p_1 - p_2))

# causal discovery accuracy
def accuracy(A_est, Gamma, n):
    return 1 - np.sum(np.abs(np.double(np.abs(A_est) > 0 ) - np.double( Gamma != 0 )))/(n*n)

# causal discovery sign accuracy
def sign_accuracy(A_est, Gamma):
    W = np.sign(A_est)*np.sign(Gamma)
    if np.sum(np.abs(A_est) > 0) > 0:
        return np.sum(W == 1)/np.sum(np.abs(A_est) > 0)
    return 0