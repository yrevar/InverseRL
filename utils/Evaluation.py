import numpy as np

def traj_log_likelihood(trajectory, s_to_idx, a_to_idx, Pi, skip_last=True):
    
    lik = 0.
    for (s, a) in trajectory[:-1 if skip_last else None]:
        s_idx, a_idx = s_to_idx[tuple(s)], a_to_idx[a]
        lik += np.log(Pi[s_idx, a_idx])
    return lik

def log_likelihood(traj_list, s_to_idx, a_to_idx, Pi, skip_last=True):
    
    lik_sum = 0.
    for traj in traj_list:
        lik_sum += traj_log_likelihood(traj, s_to_idx, a_to_idx, Pi, skip_last)
    return lik_sum

def traj_likelihood(trajectory, s_to_idx, a_to_idx, log_Pi, skip_last=True):
    
    lik = 0.
    for (s, a) in trajectory[:-1 if skip_last else None]:
        s_idx, a_idx = s_to_idx[tuple(s)], a_to_idx[a]
        lik += log_Pi[s_idx, a_idx]
    return lik

def likelihood(traj_list, s_to_idx, a_to_idx, log_Pi, skip_last=True):
    
    lik_sum = 0.
    for traj in traj_list:
        lik_sum += traj_log_likelihood(traj, s_to_idx, a_to_idx, log_Pi, skip_last)
    return lik_sum
