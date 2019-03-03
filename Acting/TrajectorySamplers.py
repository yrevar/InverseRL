import numpy as np

def sample_trajectory(S, A, T, start_state, policy, given_goal, 
                      horizon=1000, greedy_selection=True):
    
    s_list = []
    a_list = []
    
    # state tuple -> idx
    s_to_idx = {tuple(v):k for k,v in enumerate(S)}
    given_goal_idx = s_to_idx[tuple(given_goal)]
    steps = 0
    
    ## start state
    s = start_state
    
    while steps < horizon:
        
        ## add state
        s_idx = s_to_idx[tuple(s)]
        s_list.append(S[s_idx])
        
        ## sample next state
        
        # policy  (Note: taking exp because the policy is log softmax)
        Pi_s = torch.exp(policy[s_idx]).detach().numpy()
        # action selection
        if greedy_selection:
            a_idx = int(Pi_s.argmax())
        else:
            a_idx = int(np.random.choice(len(A), p=Pi_s))
        
        a_list.append(A[a_idx])
        s = T(S[s_idx], A[a_idx])
        
        steps += 1
        
        # check if goal is given and reached
        if given_goal_idx is not None and s_idx == given_goal_idx:
            break
            
    return  s_list, a_list

def sample_trajectories(N, S, A, T, start_states, policy, given_goal, 
                        horizon=1000, greedy_selection=True):
    
    traj_list = []
    
    for i in range(N):
        
        s_list, a_list = sample_trajectory(S, A, T, start_states[i], policy, given_goal, horizon, greedy_selection)
        traj_list.append((s_list, a_list))
        
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
