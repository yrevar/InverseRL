import sys
import time
import numpy as np

# Torch
import torch

# Planning
sys.path.append("../Planning/")
import Planners as Planners


"""
Many tricky parts of this implementation are highly inspired by Matthew Alger's code:
https://github.com/MatthewJA/Inverse-Reinforcement-Learning/blob/master/irl/deep_maxent.py
"""

def softmax(x1, x2):

    max_x = max(x1, x2)
    min_x = min(x1, x2)
    return max_x + np.log(1 + np.exp(min_x - max_x))

def compute_feature_expectations(trajectory_list, phi):
    
    n_sa = 0
    for idx, trajectory in enumerate(trajectory_list):
        
        # Iterate over states of trajectory.
        for (s,a) in trajectory[:-1]:
            s_idx = s_to_idx[tuple(s)]
            a_idx = a_to_idx[a]
            if n_sa == 0:
                feature_exp = phi(s).clone()
            else:
                feature_exp += phi(s)
            n_sa += 1
    return feature_exp / n_sa

def compute_svf(trajectory_list, S):
    
    s_to_idx = {tuple(v):k for k,v in enumerate(S)}
    svf = np.zeros(len(S))
    n_sa = 0
    for idx, trajectory in enumerate(trajectory_list):
        
        # Iterate over states of trajectory.
        for (s,a) in trajectory:
            svf[s_to_idx[tuple(s)]] += 1
            n_sa += 1
            
    # Note: This is deviation from Matthew Alger's code where svf is 
    # normalized by len(trajectory_list)
    return svf / n_sa 

def backward_pass(S, A, R, T, n_iters, goal, convergence_eps=1e-6, 
                           verbose=False, dtype=np.float32, gamma=0.90):
    
    # Forward Pass
    
    nS, nA = len(S), len(A)
    s_to_idx = {tuple(v):k for k,v in enumerate(S)}
    a_to_idx = {a:i for i,a in enumerate(A)}
    v_delta_max = float("inf")
    
    Q = np.zeros((nS, nA), dtype=dtype)
    Pi = np.ones((nS, nA), dtype=dtype) / nA
    #V = np.nan_to_num(np.ones((nS), dtype=dtype) * float("-inf")) # log (exp(-inf)) -> divide by zero
    V = np.nan_to_num(np.ones((nS), dtype=dtype) * -1e2)
    R = np.array([r.item() for r in R])
        
    # Given goal
    goal_state_idx = s_to_idx[tuple(goal)]
    V[goal_state_idx] = 0
    
    if verbose: print("Running Backward Pass  [ ", end="")
    iterno = 0
    while iterno < n_iters and v_delta_max > convergence_eps:
        
        if verbose and iterno and iterno % 30 == 0: 
            print(".", end="" if iterno % 300 else "\n\t")
        
        v_delta_max = 0
        for si, s in enumerate(S):
            
            v_s_prev = V[si]
            if si == goal_state_idx or s.is_terminal():
                continue

            for ai, a in enumerate(A):
                
                s_prime = T(s,a)
                if s_prime is None: # outside envelope
                    continue
                Q[si, ai] = R[si] + gamma * V[s_to_idx[tuple(s_prime)]]
                
            # Note: 
            V[si] = np.log(np.exp(Q[si,:]).sum())
            v_delta_max = max(abs(v_s_prev - V[si]), v_delta_max)
        
        iterno += 1
    
    if iterno == n_iters:
        if verbose: print(" ] Backward pass didn't converge by {}.".format(iterno))
    else:
        if verbose: print(" ] Backward pass converged @ {}.".format(iterno))
            
    # Compute softmax policy
    for si, s in enumerate(S):
        if goal_state_idx != si:
            Pi[si, :] = np.exp(Q[si,:]-V[si])
            
    return Pi, V, Q, s_to_idx, a_to_idx

def compute_expected_svf(data, S, A, Pi, T, goal, dtype=np.float32):
    
    # Forward Pass
    nS, nA = len(S), len(A)
    s_to_idx = {tuple(v):k for k,v in enumerate(S)}
    a_to_idx = {a:i for i,a in enumerate(A)}
    N = max([len(traj) for traj in data])
    
    # Given goal
    goal_state_idx = s_to_idx[tuple(goal)]
    
    # Initial visitation count
    D = np.zeros((N, nS), dtype=dtype)
    for s_idx in [s_to_idx[tuple(traj[0][0])] for traj in data]:
        D[0, s_idx] += 1.
    D[0, :] /= len(data)
    
    # N-step visitation count under given Policy and given Dynamics with intial mass distribution D[0, :]
    for n in range(N-1):
        for s_prev_idx, s_prev in enumerate(S):
            for a_idx, a in enumerate(A):
                # Note: This implementation assumes deterministic dynamics.
                s_idx = s_to_idx[tuple(T(s_prev,a))]
                p_sprev_a_s = 1.
                D[n+1, s_idx] += p_sprev_a_s * Pi[s_prev_idx, a_idx] * D[n, s_prev_idx]
    return D.sum(axis=0)

def MaxEntIRL(data, states_generator_fn, dynamics_generator_fn, 
          A, phi, R_model, R_optimizer, gamma, 
          n_iters=20, max_vi_iters=100, max_likelihood=-np.log(0.99), vi_convergence_eps=0.001, 
          dtype=torch.float32, verbose=True, print_interval=1):

    if verbose: print("{} params \n-----"
                      "\n\t Domains: {}, sizes: {},"
                      "\n\t Action dim: {}, \n\t Feature dim: {},"
                      "\n\t Iterations: {}, \n\t Max likelihood: {},"
                      "\n\t VI iterations: {}, \n\t VI convergence eps: {},"
                      "\n\t Gamma (discount factor): {},".format(
                          sys._getframe().f_code.co_name,
                          len(data), [len(states_generator_fn(traj)) for traj in data], 
                          len(A), len(phi(states_generator_fn(data[0])[0])), 
                          n_iters, np.exp(-max_likelihood), max_vi_iters, 
                          vi_convergence_eps, gamma, torch.linspace(0,1,4)))
    loss_history = []
    
    try:
        for _iter in range(n_iters):

            # mlirl iter tick
            _iter_start = time.time()

            # Zero grads
            R_optimizer.zero_grad()

            loss = 0
            n_sa = 0
            for idx, trajectory in enumerate(data):

                goal = trajectory[-1][0]
                S = states_generator_fn(trajectory)
                T = dynamics_generator_fn(trajectory)
                # torch.tensor is tempting here, but it won't pass gradients to R_model
                R = [R_model(phi(s)).type(dtype)[0] for s in S] 

                mu_expert = compute_svf(data, S)
                
                # Policy Computation.
                # Compute Policy (Backward Pass)
                Pi, V, Q, s_to_idx, a_to_idx = backward_pass(S, A, R, T, max_vi_iters, goal, 
                                                             vi_convergence_eps, verbose=True, gamma=gamma)
                
                # Policy Evaluation.
                # Forward Pass (state visitation frequency).
                mu_learner_exp = compute_expected_svf(data, S, A, Pi, T, goal)
                grad_r_s = torch.tensor(mu_expert - mu_learner_exp, dtype=dtype) # gradient for r(s) for each s in S
                
                # Compute gradient
                for i, r in enumerate(R):
                    r.backward(gradient=grad_r_s) # like scaling the identity gradient
                
            # Loss is computed per state which is equal to difference in state visitation frequency of 
            # expert and learner policies.
            # To get a single scalar loss value, I'm using norm of these gradients.
            loss = np.linalg.norm(grad_r_s, ord=1) / len(S)
            loss_history.append(loss)
            # Gradient step
            R_optimizer.step()

            if verbose and (_iter % print_interval == 0 or _iter == n_iters-1):
                print("\n>>> Iter: {:04d}: loss = {:09.6f}, CPU time = {:f}".format(
                    _iter, loss, time.time()-_iter_start))

            if max_likelihood is not None and loss < max_likelihood:
                print("\n>>> Iter: {:04d} Converged.\n\n".format(_iter))
                break
                
    except KeyboardInterrupt:
        return loss_history
    except:
        raise
    return loss_history
