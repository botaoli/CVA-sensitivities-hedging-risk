import numpy as np
from learning.cva_estimator_portfolio_int import CVAEstimatorPortfolioInt
from simulation.new_val_diffusion_engine import DiffusionEngine
import torch
import pickle
import sys
import argparse


parser = argparse.ArgumentParser(description='CVA calculation.')
parser.add_argument('--product', '-p', type=str, default='swap')
parser.add_argument('--timescale', '-t', type=str, default='d', choices=['d', 'm', 'y'])
parser.add_argument('--method', '-m', type=str, default='nested', choices=['nested', 'linear', 'oneshot_NN', 'transferred_NN'])
parser.add_argument('--out_path', '-o', type=str, default='output')
parser.add_argument('--device', '-d', type=int, default=0)
parser.add_argument('--test', action='store_true')
parser.add_argument('--n_rate', type=int, default=10)
parser.add_argument('--n_spread', type=int, default=9)

args = parser.parse_args()
print(args)

np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.benchmark = False # don't allow cudnn to tune for every input size
torch.backends.cudnn.enabled = True

dict_coarse = {'d':1000, 'm':100, 'y':10}
dict_fine_coarse = {'d':1, 'm':25, 'y':25}
year_list = [1, 2, 3, 4, 8, 9]

num_coarse_steps = dict_coarse[args.timescale]
dT = 10.0 / num_coarse_steps
num_fine_per_coarse = dict_fine_coarse[args.timescale]
dt = dT/num_fine_per_coarse
if num_coarse_steps == 10:
    step_list = year_list
else:
    step_list = [int(year * num_coarse_steps / 10) for year in year_list]
    step_list = [1] + step_list

num_paths = 50000
num_inner_paths = 1
num_defs_per_path = 256
if args.method == 'nested':
    num_inner_paths = 512+2
if args.test:
    num_paths = 16
    num_inner_paths = 8 + 2
    num_defs_per_path = 8    

num_rates = args.n_rate
num_spreads = args.n_spread
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
num_vanillas = 500 * (1 - int(args.product == 'swap'))
vanilla_specs = np.empty(num_vanillas,
                         dtype=[('maturity', '<f4'), ('notional', '<f4'),
                                ('strike', '<f4'), ('cpty', '<i4'),
                                ('undl', '<i4'), ('call_put', '<b1')])
vanilla_specs['maturity'] = np.random.uniform(0.1,9.5,num_vanillas)
vanilla_specs['notional'] = 1000. * ((np.random.choice((-1, 1), num_vanillas, p=(0.5, 0.5)))
     * np.random.choice(range(1, 6), num_vanillas))
vanilla_specs['strike'] = abs(1. * np.random.uniform(0.9,1.1, num_vanillas))
vanilla_specs['cpty'] = np.random.randint(0, num_spreads-1, num_vanillas, np.int32)  # Counterparty with which the swap was entered into
vanilla_specs['undl'] = np.random.randint(0, num_rates-1, num_vanillas, np.int32)  # Underlying currency
vanilla_specs['call_put'] = np.random.choice((True, False), num_vanillas, p=(0.5, 0.5))

num_irs = 500 * int(args.product == 'swap')
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

out_name = 'test'
if not args.test:
    out_name = args.product + '1' + args.timescale + '_' + args.method
device = torch.device('cuda:{}'.format(args.device))
print(device)

print(num_coarse_steps, dT, num_fine_per_coarse, dt, num_vanillas, num_irs, out_name)
diffusion_engine = DiffusionEngine(50, 50, num_coarse_steps, dT, num_fine_per_coarse, dt,
                                   num_paths, num_inner_paths, num_defs_per_path, 
                                   num_rates, num_spreads, R, rates_params, fx_params, 
                                   spreads_params, vanilla_specs, irs_specs, zcs_specs,
                                   initial_values, initial_defaults, cDtoH_freq, device.index, nb_nested=len(step_list))

# selector for previous swap resets, need the states at those dates because of how the small non-Markovianity in the swap prices
#reset_period = max(1, int(2 * num_coarse_steps / 100))
#prev_reset_arr = (np.arange(num_coarse_steps+1)-1)//reset_period*reset_period
prev_reset_arr = (np.arange(num_coarse_steps+1)-1)//2*2
print(prev_reset_arr)

# learner with default intensities
cva_estimator_portfolio_int = CVAEstimatorPortfolioInt(prev_reset_arr, True, False, False, diffusion_engine, 
                                                       device, 1, 2*(num_rates+num_spreads), (num_defs_per_path*num_paths)//32, 
                                                       1, 0.01, 0, reset_weights=False, linear=False, best_sol=True)

if args.method == 'nested':
    diffusion_engine.generate_batch(fused=True, nested_cva_at=step_list, indicator_in_cva=False)
else:
    diffusion_engine.generate_batch(fused=True)
    if args.method == 'linear':
        cva_estimator_portfolio_int = CVAEstimatorPortfolioInt(prev_reset_arr, True, False, False, diffusion_engine, 
                                                               device, 1, 2*(num_rates+num_spreads), (num_defs_per_path*num_paths)//32, 
                                                               100, 0.01, 0, reset_weights=False, linear=True, best_sol=True, no_nested_cva = True)
    elif args.method == 'oneshot_NN':
        cva_estimator_portfolio_int = CVAEstimatorPortfolioInt(prev_reset_arr, True, False, False, diffusion_engine, 
                                                               device, 1, 2*(num_rates+num_spreads), (num_defs_per_path*num_paths)//32, 
                                                               2000, 0.01, 0, reset_weights=False, linear=False, best_sol=True, no_nested_cva = True)
    elif args.method == 'transferred_NN':
        cva_estimator_portfolio_int = CVAEstimatorPortfolioInt(prev_reset_arr, True, False, False, diffusion_engine, 
                                                               device, 1, 2*(num_rates+num_spreads), (num_defs_per_path*num_paths)//32, 
                                                               16, 0.01, 0, reset_weights=False, linear=False, best_sol=True, no_nested_cva = True)

nested_CVA = diffusion_engine.nested_cva

cva_portfolio_features_gen_int = cva_estimator_portfolio_int._build_features()
if not args.method == 'nested':
    if args.method == 'transferred_NN':
        cva_estimator_portfolio_int.train(features_gen=cva_portfolio_features_gen_int, labels_as_cuda_tensors=True, train_time = None)
    else:
        cva_estimator_portfolio_int.train(features_gen=cva_portfolio_features_gen_int, labels_as_cuda_tensors=True, train_time = step_list)

label_gen = cva_estimator_portfolio_int._build_labels()
labels = {}
for i, label in enumerate(label_gen):
    i_step = num_coarse_steps - i
    if i_step in step_list:
        labels[i_step] = label.copy()
    
y0_CVA = diffusion_engine.nested_cva_save1
y1_CVA = diffusion_engine.nested_cva_save2

out_file = './{}/{}.pickle'.format(args.out_path, out_name)
if args.test:
    out_file = 'test.pickle'

with open(out_file, 'wb') as f:
    data = {}
    for index, step in enumerate(step_list): 
        data['X{}'.format(step)] = diffusion_engine.X[step]
        if args.method == 'nested':
            data['nested_CVA_{}'.format(step)] = nested_CVA[index]
            data['y0_CVA_{}'.format(step)] = y0_CVA[index]
            data['y1_CVA_{}'.format(step)] = y1_CVA[index]
        next(cva_portfolio_features_gen_int)
        data['feature{}'.format(step)] = cva_portfolio_features_gen_int.send(step).copy()
        data['label{}'.format(step)] = labels[step]
    if not args.method == 'nested':
        data['model'] = cva_estimator_portfolio_int.saved_states
    data['mtm'] = diffusion_engine.mtm_by_cpty
    pickle.dump(data, f)
