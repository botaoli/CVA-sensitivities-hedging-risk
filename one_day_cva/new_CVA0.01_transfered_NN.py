import matplotlib.pyplot as plt
import numpy as np
from learning.cva_estimator_portfolio_int import CVAEstimatorPortfolioInt
from learning.cva_estimator_portfolio_def import CVAEstimatorPortfolioDef
from simulation.diffusion_engine import DiffusionEngine
import time
import torch
import matplotlib as mpl
import pickle

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False # don't allow cudnn to tune for every input size
torch.backends.cudnn.enabled = True

device = torch.device('cuda:0')

num_coarse_steps = 1000
dT = 0.01
num_fine_per_coarse = 25
dt = 0.0004
num_paths = 100000
num_inner_paths = 1024
num_defs_per_path = 64
num_rates = 10
num_spreads = 9
R = np.eye(2*num_rates-1+num_spreads, dtype=np.float32) # we set the correlation matrix to the identity matrix, although not needed
initial_values = np.empty(2*num_rates-1+num_spreads, dtype=np.float32)
initial_defaults = np.empty((num_spreads-1+7)//8, dtype=np.int8)

# rates diffusion parameters
rates_params = np.empty(num_rates, dtype=[('a', '<f4'), ('b', '<f4'), ('sigma', '<f4')])
rates_params['a'] = np.random.normal(0.4, 0.03, num_rates)
rates_params['b'] = np.random.normal(0.03, 0.001, num_rates)
rates_params['sigma'] = np.abs(np.random.normal(0.0025, 0.00025, num_rates))
initial_values[:num_rates] = 0.01

# FX diffusion parameters
fx_params = np.empty(num_rates-1, dtype=[('vol', '<f4')])
fx_params['vol'] = np.abs(np.random.normal(0.25, 0.025, num_rates-1))
initial_values[num_rates:2*num_rates-1] = 1

# stochastic intensities diffusion parameters
spreads_params = np.empty(num_spreads, dtype=[('a', '<f4'), ('b', '<f4'), ('vvol', '<f4')])
spreads_params['a'] = np.random.normal(0.5, 0.03, num_spreads)
spreads_params['b'] = np.random.normal(0.01, 0.001, num_spreads)
spreads_params['vvol'] = np.abs(np.random.normal(0.0075, 0.00075, num_spreads))
initial_values[2*num_rates-1:] = 0.01

# initial default indicators
initial_defaults[:] = 0

# length of simulated path on the GPU (paths are then simulated by chunks of cDtoH_freq until maturity)
cDtoH_freq = 20

# product specs (DO NOT use the ZCs)
num_vanillas = 0
vanilla_specs = np.empty(num_vanillas,
                         dtype=[('maturity', '<f4'), ('notional', '<f4'),
                                ('strike', '<f4'), ('cpty', '<i4'),
                                ('undl', '<i4'), ('call_put', '<b1')])

num_irs = 500
irs_specs = np.empty(num_irs,
                     dtype=[('first_reset', '<f4'), ('reset_freq', '<f4'),
                            ('notional', '<f4'), ('swap_rate', '<f4'),
                            ('num_resets', '<i4'), ('cpty', '<i4'),
                            ('undl', '<i4')])

irs_specs['first_reset'] = 0.  # First reset date in the swaps
irs_specs['reset_freq'] = 0.2  # Reset frequency
irs_specs['notional'] = 10000. * \
    ((np.random.choice((-1, 1), num_irs, p=(0.5, 0.5)))
     * np.random.choice(range(1, 11), num_irs))  # Notional of the swaps
irs_specs['swap_rate'] = np.abs(np.random.normal(0.03, 0.001, num_irs))  # Swap rate, not needed, swaps are priced at par anyway
irs_specs['num_resets'] = np.random.randint(6, 51, num_irs, np.int32)  # Number of resets (num_resets*reset_freq should be equal to the desired maturity)
irs_specs['cpty'] = np.random.randint(0, num_spreads-1, num_irs, np.int32)  # Counterparty with which the swap was entered into
irs_specs['undl'] = np.random.randint(0, num_rates-1, num_irs, np.int32)  # Underlying currency

num_zcs = 0
zcs_specs = np.empty(num_zcs,
                     dtype=[('maturity', '<f4'), ('notional', '<f4'),
                            ('cpty', '<i4'), ('undl', '<i4')])

diffusion_engine = DiffusionEngine(50, 50, num_coarse_steps, dT, num_fine_per_coarse, dt,
                                   num_paths, num_inner_paths, num_defs_per_path, 
                                   num_rates, num_spreads, R, rates_params, fx_params, 
                                   spreads_params, vanilla_specs, irs_specs, zcs_specs,
                                   initial_values, initial_defaults, cDtoH_freq, device.index)

# selector for previous swap resets, need the states at those dates because of how the small non-Markovianity in the swap prices
prev_reset_arr = (np.arange(num_coarse_steps+1)-1)//2*2


# learner with default intensities
cva_estimator_portfolio_int = CVAEstimatorPortfolioInt(prev_reset_arr, True, False, False, diffusion_engine, 
                                                       device, 1, 2*(num_rates+num_spreads), (num_defs_per_path*num_paths)//32, 
                                                       16, 0.01, 0, reset_weights=False, linear=False, best_sol=True, no_nested_cva = True)

diffusion_engine.generate_batch(fused=True)

cva_portfolio_features_gen_int = cva_estimator_portfolio_int._build_features()
print('Training using default intensities...', end='')
_chrono_start = time.time()
cva_estimator_portfolio_int.train(features_gen=cva_portfolio_features_gen_int, labels_as_cuda_tensors=True, train_time = None)
_chrono_end = time.time()
elapsed_time_def = _chrono_end - _chrono_start
print('done in {} secs'.format(round(elapsed_time_def, 2)))

'''
X_int = np.empty((2, num_defs_per_path*num_paths,8), dtype=np.float32)
_v_ = X_int.reshape((2, num_paths* num_defs_per_path,8))
for i,t in enumerate([0,1]):
    next(cva_portfolio_features_gen_int)
    _v_[i]= cva_portfolio_features_gen_int.send(t)[:,37:]#.clone().cpu().numpy()

Xfull_int = np.empty((2, num_defs_per_path*num_paths,36), dtype=np.float32)
for t in range(2):
    Xfull_int[t] = np.concatenate([np.tile(diffusion_engine.X[t], num_defs_per_path).T,
                    X_int[t]],axis = 1)

cva_portfolio_predictor_int = cva_estimator_portfolio_int.predict(features_gen=cva_portfolio_features_gen_int, as_cuda_array=True, flatten=False)
predicted_cva_portfolio_out_int = np.empty((num_coarse_steps+1, num_defs_per_path*num_paths), dtype=np.float32)
_v_int = predicted_cva_portfolio_out_int.reshape((num_coarse_steps+1, num_defs_per_path, num_paths))
for t in range(num_coarse_steps, -1, -1):        
    next(cva_portfolio_predictor_int)
    _v_int[t] = cva_portfolio_predictor_int.send(t)
_chrono_end = time.time()

nested_CVA_1 = diffusion_engine.nested_cva[1].reshape(-1,1)
'''


with open('./new_CVA0.01_transfered_NN.pickle', 'wb') as f:
        data = {}
        '''
        data["X0"] = Xfull_int[0][0:1]
        data["X1"] = Xfull_int[1]
        data["CVA0"] = predicted_cva_portfolio_out_int[0][0:1] 
        data["CVA1"] = predicted_cva_portfolio_out_int[1]
        '''
        data['model'] = cva_estimator_portfolio_int.saved_states
        data['time'] = _chrono_end - _chrono_start
        pickle.dump(data, f)