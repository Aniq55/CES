import numpy as np 
import pickle
from utils import *
from metrics import *
from tqdm import tqdm


ATD_logger = create_logger('info_logger', '/home/chri6578/Documents/CES/logs/ATD.log', logging.INFO)
ACD_logger = create_logger('warning_logger', '/home/chri6578/Documents/CES/logs/ACD.log', logging.WARNING)
OD_logger = create_logger('error_logger', '/home/chri6578/Documents/CES/logs/OD.log', logging.ERROR)

param_name = "Theta_1"

with open(f'/home/chri6578/Documents/CES/params/{param_name}.pickle', 'rb') as file:
    Theta = pickle.load(file)

n_iters = int(1e3)

for iter_ in tqdm(range(n_iters)):
    E_1 = ESG(Theta)
    E_2 = ESG(Theta)
    
    atd = ATD(E_1, E_2)
    # LOG: atd
    ATD_logger.info(f'{atd}')
    
    for v in np.arange(-2,5,0.5):
        tau = np.power(10, v)
        acd = ACD(E_1, E_2, tau)
        # LOG: v, acd
        ACD_logger.warning(f'{v}, {acd}')
        
    for k in range(2,6):
        od = OD(E_1, E_2, k)
        # LOG: k, od
        OD_logger.error(f'{k}, {od}')




