import torch
import numpy as np

def sample_trajectory(S, A, T, start_state, policy, given_goal, 
                      horizon=1000, greedy_selection=True, values=None):
    
    trajectory = []
    
    # state tuple -> idx
    s_to_idx = {v:k for k,v in enumerate(S)}
    given_goal_idx = s_to_idx[S.at_loc(given_goal)]
    steps = 0
    
    ## start state

    s = S.at_loc(start_state)

    while steps < horizon:
        
        ## add state
        s_idx = s_to_idx[s]
        
        if given_goal_idx == s_idx:
            trajectory.append((s, None))
            break
            
        ## sample next state
        
        # policy  (Note: taking exp because the policy is log softmax)
        Pr_s = np.exp(policy[s_idx])
        # action selection
        if greedy_selection:
            a_idx = int(Pr_s.argmax())
        else:
            if values is not None:
                v_curr = values[s_idx].item()
                # To prevent suboptimal trajectories with cycles
                S_primes = [T(S[s_idx], A[a_idx]) for a_idx in range(len(A))]
                V_primes = np.asarray([values[s_to_idx[sp]].item() for sp in S_primes])
                V_improves = V_primes >= v_curr
                
                if sum(V_improves) == 0:
                    raise Exception("Passed value function isn't converged or the goal is incorrect.")
                
                Pr_s[~V_improves] = 0. # set prob. mass of not value improving actions to 0.
                Pr_s = Pr_s / Pr_s.sum() # normalize s.t. probs sum to 1.
            a_idx = int(np.random.choice(len(A), p=Pr_s))
            
        trajectory.append((S[s_idx], A[a_idx]))
        s_prime_lst, s_prob = list(zip(*T(S[s_idx], A[a_idx])))
        slct_s_prime_idx = np.random.choice(np.arange(len(s_prob)), p=s_prob)
        s = s_prime = s_prime_lst[slct_s_prime_idx]
        
        steps += 1
        
        # check if goal is given and reached
        if given_goal_idx is not None and s_idx == given_goal_idx:
            break
            
    return  trajectory

def sample_trajectories(N, S, A, T, start_states, policy, given_goal, 
                        horizon=1000, greedy_selection=True, values=None):
    
    traj_list = []
    
    for i in range(N):
        
        trajectory = sample_trajectory(S, A, T, start_states[i], policy, given_goal, horizon, greedy_selection, values)
        traj_list.append(trajectory)
        
    return traj_list

def get_min_value_estimate_given_goal_greedy(max_traj_length, driving_cost, gamma):
    
    min_value = 0.
    for i in range(max_traj_length):
        min_value -= (gamma**i) * driving_cost
    return min_value


def sample_shortest_path_trajectories(start_states, goal_states, shortest_path_fn):
    
    traj_list = []
    for i in range(len(start_states)):
        start = start_states[i]
        goal = goal_states[i]
        traj_list.append(shortest_path_fn(start, goal))
    return traj_list 
