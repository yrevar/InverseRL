import sys
import time
import numpy as np

# Torch
import torch

# Planning
sys.path.append("../Planning/")
import Planners as Planners

def MLIRL(data, states_generator_fn, dynamics_generator_fn, 
          A, phi, R_model, R_optimizer, policy, gamma, 
          n_iters=20, max_vi_iters=100, max_likelihood=-np.log(0.99), vi_convergence_eps=0.001, 
          dtype=torch.float32, verbose=True, print_interval=1):

    if verbose: print("MLIRL params \n-----"
                      "\n\t Domains: {}, sizes: {},"
                      "\n\t Action dim: {}, \n\t Feature dim: {},"
                      "\n\t Iterations: {}, \n\t Max likelihood: {},"
                      "\n\t VI iterations: {}, \n\t VI convergence eps: {},"
                      "\n\t Gamma (discount factor): {},"
                      "\n\t Policy example: Q {} -> Pi {}".format(
                          len(data), [len(states_generator_fn(traj)) for traj in data], 
                          len(A), len(phi(states_generator_fn(data[0])[0])), 
                          n_iters, np.exp(-max_likelihood), max_vi_iters, 
                          vi_convergence_eps, gamma, torch.linspace(0,1,4), policy(torch.linspace(0,1,4))))
    loss_history = []
    
    try:
        for _iter in range(n_iters):

            # mlirl iter tick
            _mlirl_iter_start = time.time()

            # Zero grads
            R_optimizer.zero_grad()

            loss = 0
            n_sa = 0
            for idx, trajectory in enumerate(data):

                goal = trajectory[-1][0]
                S = states_generator_fn(trajectory)
                T = dynamics_generator_fn(trajectory)
                # torch.tensor is tempting here, but it won't pass gradients to R_model
                R = [R_model(phi(s)).type(dtype) for s in S] 

                # Compute Policy
                log_Pi, V, Q, s_to_idx, a_to_idx = Planners.differentiable_value_iteration(
                    S, A, R, T, policy, gamma, max_vi_iters, goal, 
                    convergence_eps=vi_convergence_eps, detach_R=False, verbose=False, dtype=dtype)

                # Maximize data likelihood objective
                for (s,a) in trajectory[:-1]:
                    s_idx = s_to_idx[tuple(s)]
                    a_idx = a_to_idx[a]
                    loss -= log_Pi[s_idx, a_idx]
                    n_sa += 1
            
            # Normalize to make likelihood independent of trajectory length.
            loss = (1./n_sa) * loss
            loss.backward()
            loss_history.append(loss.detach().item())
            # Gradient step
            R_optimizer.step()

            if verbose and (_iter % print_interval == 0 or _iter == n_iters-1):
                print("\n>>> Iter: {:04d}: loss = {:09.6f}, likelihood = {:02.4f}, CPU time = {:f}".format(
                    _iter, loss, np.exp(-loss.item()), time.time()-_mlirl_iter_start))

            if max_likelihood is not None and loss < max_likelihood:
                print("\n>>> Iter: {:04d} Converged.\n\n".format(_iter))
                break
                
    except KeyboardInterrupt:
        return loss_history
    except:
        raise
    return loss_history