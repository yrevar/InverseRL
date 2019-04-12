import numpy as np

def traj_log_likelihood(trajectory, s_to_idx, a_to_idx, Pi, skip_last=True):
    """
    Computes log likelihood of given trajectory for a given policy.
    """
    lik = 0.
    for (s, a) in trajectory[:-1 if skip_last else None]:
        s_idx, a_idx = s_to_idx[s], a_to_idx[a]
        lik += np.log(Pi[s_idx, a_idx]) # Take log
    return lik

def traj_likelihood(trajectory, s_to_idx, a_to_idx, log_Pi, skip_last=True):
    """
    Computes likelihood of given trajectory for a given log policy.
    """
    lik = 0.
    for (s, a) in trajectory[:-1 if skip_last else None]:
        s_idx, a_idx = s_to_idx[s], a_to_idx[a]
        lik += log_Pi[s_idx, a_idx] # No need to take log
    return lik

def log_likelihood(traj_list, s_to_idx, a_to_idx, Pi, skip_last=True):
    """
    Computes log likelihood of given data for a given policy.
    """
    lik_sum = 0.
    for traj in traj_list:
        # Compute log likelihood
        lik_sum += traj_log_likelihood(traj, s_to_idx, a_to_idx, Pi, skip_last)
    return lik_sum

def likelihood(traj_list, s_to_idx, a_to_idx, log_Pi, skip_last=True):
    """
    Computes likelihood of given data for a given log policy.
    """
    lik_sum = 0.
    for traj in traj_list:
        # Compute likelihood
        lik_sum += traj_likelihood(traj, s_to_idx, a_to_idx, log_Pi, skip_last)
    return lik_sum
