import numpy as np 
from utils import *

# Average Time Difference
def ATD_half(E_1, E_2):
    atd = 0.0
    for e in E_1.keys():
        sum_e = 0.0
        for t in E_1[e]:
            sum_e += np.min(np.abs(t - E_2[e]))
        atd += sum_e / len(E_1)
    
    return atd/len(E_1.keys())


def ATD(E_1, E_2):
    return ATD_half(E_1, E_2) + ATD_half(E_2, E_1)


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