import numpy as np
from collections import defaultdict

# Torch
import torch
from torch import nn

# Utils
import InverseRL.utils.PriorityQueue as PriorityQueue

boltzmann_policy = lambda Q, boltzmann_temp: log_boltzmann_dist(Q, boltzmann_temp)
heuristic_l2 = lambda s1, s2: np.linalg.norm(np.array(s1) - np.array(s2))

def log_boltzmann_dist(Q, temperature):
    """
    PyTorch softmax implementation seems stable, but log of softmax is not. 
    So log of boltzmann distribution is used.
    PyTorch Softmax Note:
        This function doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use log_softmax instead (it's faster and has better numerical properties).
    """
    return nn.LogSoftmax(dim=0)(Q/temperature)

def value_iteration(S, A, R, T, policy, gamma, n_iters, goal, convergence_eps=1e-3, 
                    verbose=False, dtype=torch.float32):

    # assert torch.all(R < 0)
    nS, nA = len(S), len(A)
    v_delta_max = float("inf")
    s_to_idx = {v:k for k,v in enumerate(S)}
    a_to_idx = {a:i for i,a in enumerate(A)}
    
    Q = torch.zeros(nS, nA, dtype=dtype)
    log_Pi = torch.log(torch.ones(nS, nA, dtype=dtype) / nA)
    # R can be list of differentiable functions, or a differentiable vector.
    # Intialize V. (Can't make differentiable because it'll be overwritten)
    V = torch.tensor([r for r in R], requires_grad=False)
        
    # Given goal
    goal_state_idx = s_to_idx[goal]
    V[goal_state_idx] = torch.tensor(0)
    
    if verbose: print("Running VI [ ", end="")
    iterno = 0
    while iterno < n_iters and v_delta_max > convergence_eps:
        
        if verbose and iterno and iterno % 30 == 0: 
            print(".", end="" if iterno % 300 else "\n\t")
        
        v_delta_max = 0
        for si, s in enumerate(S):
            
            v_s_prev = V[si].detach().item()
            if si == goal_state_idx or s.is_terminal():
                continue
            
            for ai, a in enumerate(A):
                
                for s_prime, p in T(s,a):
                
                    if s_prime is None: # outside envelope
                        continue
                    Q[si, ai] = R[si] + gamma * p * V[s_to_idx[s_prime]].clone()
                    
            # Softmax action selection
            log_Pi[si, :] = policy(Q[si, :].clone())
            # Softmax value
            V[si] = torch.exp(log_Pi[si, :].clone()).dot(Q[si, :].clone())
            v_delta_max = max(abs(v_s_prev - V[si].detach().item()), v_delta_max)
            
        iterno += 1
        
    if iterno == n_iters:
        if verbose: print(" ] VI didn't converge by {}.".format(iterno))
    else:
        if verbose: print(" ] VI converged @ {}.".format(iterno))
        
    return log_Pi, V, Q, s_to_idx, a_to_idx

def astar_find_path(start, goal, actions, trans_func, cost_fn=lambda s: 1, heuristic_fn=heuristic_l2):
    
    frontier = PriorityQueue.PriorityQueue()
    frontier.append([cost_fn(start)+heuristic_fn(start, goal), cost_fn(start), start])
    explored = defaultdict(lambda: False)
    cost = defaultdict(lambda: np.float("inf"))
    parent = defaultdict(lambda: None)
    path = []
    
    while frontier.size():
        
        f, g, s = frontier.pop()
        explored[s] = True
        
        if s == goal:
            
            curr_state = goal
            s_list = [curr_state]
            a_list = [None]
            parent_state = parent[curr_state] # greedy selection
            
            while parent_state is not None:
                
                s_list.append(parent_state)
                a_list.append(get_action_from_state_change(parent_state, curr_state))
                
                curr_state = parent_state
                parent_state = parent[curr_state] # greedy selection
                
            return list(zip(s_list[::-1], a_list[::-1]))
        
        for a in actions:
            
            sp = trans_func(s, a)
            if not explored[sp] and sp not in frontier:
                
                g_new = g + cost_fn(sp)
                f_new = g_new + heuristic_fn(sp, goal)
                
                # prevent cycles
                if g_new < cost[sp]:
                    frontier.append((f_new, g_new, sp))
                    parent[sp] = s
                    cost[sp] = g_new
    return None


def get_action_from_state_change(state, n_state):
    
    dx = n_state[0] - state[0]
    dy = n_state[1] - state[1]
    if dx > 0 and dy == 0: return "right"
    elif dx > 0 and dy > 0: return "up-right"
    elif dx == 0 and dy > 0: return "up"
    elif dx < 0 and dy > 0: return "up-left"
    elif dx < 0 and dy == 0: return "left"
    elif dx < 0 and dy < 0: return "down-left"
    elif dx == 0 and dy < 0: return "down"
    elif dx > 0 and dy < 0: return "down-right"
    else: return "stay"

