import torch


def run_value_iteration(S, A, R, T, s_to_idx, expl_policy, gamma, max_iters,
                        dtype, given_goal=None):

    nS, nA = len(S), len(A)
    # No need to compute value for terminal and given goal states.
    S_ = [s for s in S if not (
        s.is_terminal() or (given_goal is not None and s == given_goal))]
    # Initialize Pi, V, & Q.
    Pi = torch.ones(nS, nA, dtype=dtype) / nA
    V = R.clone()
    Q = R.clone().reshape(nS, 1).repeat(1, nA)
    # VI
    for _ in range(max_iters):
        for si, s in enumerate(S_):
            for ai, a in enumerate(A):
                for sp in T[s][a]:
                    Q[si, ai] = R[si].clone() + \
                        gamma * T[s][a][sp] * V[s_to_idx[sp]].clone()
            Q_s = Q[si, :].clone()
            Pi[si, :] = expl_policy(Q_s)
            V[si] = Pi[si, :].clone().dot(Q_s)
    return Pi, V, Q


def compute_policy(S, A, R, T, idx_to_s, gamma, n_iters,
                   expl_policy, dtype, given_goal_idx=None):

    nS, nA = len(S), len(A)
    # Policy
    Pi = torch.ones(nS, nA, dtype=dtype) / nA
    # Value
    V = R[:, 0].clone()
    # Q value
    Q = R.repeat(1, nA).clone()

    # Check if state is terminal (stop leaking values back to
    # non-goal state space)
    # Done here so as to improve performance.
    S_ = [si for si in S if not (
        idx_to_s[si].is_terminal() or given_goal_idx == si)]

    if given_goal_idx:
        V[given_goal_idx] = 0

    # Value iteration
    for _vi_iter in range(n_iters):
        for si in S_:
            for ai in A:
                Q[si, ai] = R[si].clone() + gamma * T[si][ai].dot(V.clone())
            Pi[si, :] = expl_policy(Q[si, :].clone())
            V[si] = Pi[si, :].clone().dot(Q[si, :].clone())
    return Pi, V, Q
