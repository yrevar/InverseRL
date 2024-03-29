import numpy as np
from collections import defaultdict

# Torch
import torch
import navigation_mdp as NvMDP

# Utils
import utils.PriorityQueue as PriorityQueue


class StateActionArray(object):

    def __init__(self, S, A, dtype=torch.float32):
        self.S = S
        self.A = A
        self.s_to_idx = {v: k for k, v in enumerate(S)}
        self.s_loc_to_idx = {v.get_location(): k for k, v in enumerate(S)}
        self.a_to_idx = {a: i for i, a in enumerate(A)}
        self.dtype = dtype
        self.SAValues = torch.zeros(len(S), len(A), dtype=self.dtype)

    def init_uniform(self):
        self.SAValues = torch.ones(len(self.S), len(self.A), dtype=self.dtype) / len(self.A)

    def init_zeros(self):
        self.SAValues = torch.zeros(len(self.S), len(self.A), dtype=self.dtype)

    def __getitem__(self, key):
        if isinstance(key[0], NvMDP.state.State) and isinstance(key[1], str):
            return self.SAValues[self.s_to_idx[key[0]], self.a_to_idx[key[1]]]
        elif isinstance(key[0], tuple) and isinstance(key[1], str):
            return self.SAValues[self.s_to_idx[self.S.at_loc(key[0])], self.a_to_idx[key[1]]]
        elif isinstance(key[0], NvMDP.state.State) and isinstance(key[1], slice):
            return self.SAValues[self.s_to_idx[key[0]], key[1]]
        elif isinstance(key[0], tuple) and isinstance(key[1], slice):
            return self.SAValues[self.s_to_idx[self.S.at_loc(key[0])], key[1]]
        elif (isinstance(key[0], int) or isinstance(key[0], slice)) and (
                isinstance(key[1], int) or isinstance(key[1], slice)):
            return self.SAValues[key[0], key[1]]
        else:
            raise ValueError("Invalid key!")

    def __setitem__(self, key, value):
        if isinstance(key[0], NvMDP.state.State) and isinstance(key[1], str):
            self.SAValues[self.s_to_idx[key[0]], self.a_to_idx[key[1]]] = value
        elif isinstance(key[0], tuple) and isinstance(key[1], str):
            self.SAValues[self.s_to_idx[self.S.at_loc(key[0])], self.a_to_idx[key[1]]] = value
        elif isinstance(key[0], NvMDP.state.State) and isinstance(key[1], slice):
            self.SAValues[self.s_to_idx[key[0]], key[1]] = value
        elif isinstance(key[0], tuple) and isinstance(key[1], slice):
            self.SAValues[self.s_to_idx[self.S.at_loc(key[0])], key[1]] = value
        elif isinstance(key[0], int) and (isinstance(key[1], int) or isinstance(key[1], slice)):
            self.SAValues[key[0], key[1]] = value
        else:
            raise ValueError("Invalid key!")

    def data(self):
        return self.SAValues

    def __repr__(self):
        return str(f'{self.__class__}\ndata: {self.data()}')


class ValueIteration:

    def __init__(self, discrete_state_space, rewards, dynamics, gamma=0.95, goal=None, goal_init_value=0., verbose=False,
                 log_pi=False, dtype=torch.float32):
        if isinstance(rewards, torch.Tensor):
            self.R = rewards.detach()
        elif isinstance(rewards, list) and isinstance(rewards[0], torch.Tensor):
            self.R = [r.detach() for r in rewards]
        else:
            self.R = rewards
        # if not (isinstance(self.R[0], np.float32) or isinstance(self.R[0], np.float64)):
        #     print(f"Warning: Reward type is {type(self.R[0])}!")
        self.R_grad = rewards
        self.T = dynamics
        self.S, self.nS = discrete_state_space, len(discrete_state_space)
        self.A, self.nA = self.T.ACTIONS, len(self.T.ACTIONS)
        self.gamma = gamma
        self.goal = goal
        self.goal_init_value = goal_init_value
        self.verbose = verbose
        # TODO: Ref: https://github.com/yrevar/InverseRL/blob/master/MLIRL/Differentiable_Value_Iteration_4_Actions.ipynb
        self.start_reasoning = False
        # self.log_pi = log_pi
        self.dtype = dtype
        self.reset()

    def reset(self):
        self.iterno = 0
        self.initialize()

    def check_goal(self):
        # reset terminal status of all states
        self.S.reset_terminal_status()
        if self.goal is not None:
            if not isinstance(self.goal, NvMDP.state.State):
                self.goal = self.S.at_loc(self.goal)
        else:
            if self.verbose: print("No specific goals")
            self.goal = None
            return True

        is_in_state_space = False
        for s in self.S:
            if s == self.goal:
                is_in_state_space = True
        if not is_in_state_space:
            raise Exception("Goal is not in state space!")

        if self.verbose: print("Setting goal {} as terminal state!".format(self.goal))
        self.goal.set_terminal_status(True)
        return True

    def initialize(self):
        self.s_to_idx = {v: k for k, v in enumerate(self.S)}
        self.a_to_idx = {a: i for i, a in enumerate(self.A)}
        self.V = torch.tensor([r for r in self.R], requires_grad=False, dtype=self.dtype)
        self.Q = StateActionArray(self.S, self.A, dtype=self.dtype)
        self.Q.init_zeros()
        self.Pi = StateActionArray(self.S, self.A, dtype=self.dtype)
        self.Pi.init_uniform()
        self.check_goal()
        if self.goal is not None:
            self.V[self.s_to_idx[self.goal]] = torch.tensor(self.goal_init_value)
        self.v_delta_max = None
        self.converged = False
        # if self.log_pi:
        #     self.Pi = torch.log(torch.ones(self.nS, self.nA, dtype=torch.float32) / self.nA)

    def get_tbl_idxs(self, state, action):
        return self.s_to_idx[state], self.a_to_idx[action]

    def q_value(self, s, a, debug=False):
        si = self.s_to_idx[s]
        q = 0
        for s_prime, p in self.T(s, a):
            if p > 0:
                # if debug: print(s, a, "- {} ->".format(p), s_prime)
                if s_prime is None:  # outside envelope
                    continue
                if debug and s_prime.is_terminal():
                    print("\n {}, {} -> {}, R[s]={:.2f}, T(s,a,s')={:.2f},  gamma {:.2f}, V'[TERM]={:.2f}, "
                          "E[V'[TERM]]={:.2f}".format(
                        s, a, s_prime, self.R[si], p, self.gamma, self.V[self.s_to_idx[s_prime]],
                        self.gamma * p * self.V[self.s_to_idx[s_prime]].clone()), end="")
                q += self.gamma * p * self.V[self.s_to_idx[s_prime]].clone()
        if self.start_reasoning:
            return self.R_grad[si] + q
        else:
            return self.R[si] + q

    def q_value_list(self, s, debug=False):
        return [self.q_value(s, a, debug) for a in self.A]

    def step(self, policy_fn, debug=False, ret_vals=False):
        v_delta_max = 0
        for si, s in enumerate(self.S):
            v_s__old = self.V[si].detach().item()
            if s.is_terminal():
                if debug: print("Terminal: {}".format(s))
                continue
            for ai, q in enumerate(self.q_value_list(s, debug=debug)):
                self.Q[si, ai] = q
            # Softmax action selection
            self.Pi[si, :] = policy_fn(self.Q[si, :].clone())
            # Softmax value
            # if self.log_pi:
            #     self.V[si] = torch.exp(self.Pi[si, :].clone()).dot(self.Q[si, :].clone())
            # else:
            self.V[si] = self.Pi[si, :].clone().dot(self.Q[si, :].clone())
            v_delta_max = max(abs(v_s__old - self.V[si].detach().item()), v_delta_max)
        self.iterno += 1
        if ret_vals:
            return self.Pi, self.V, self.Q, self.iterno, v_delta_max
        else:
            return v_delta_max

    def run(self, max_iters, policy_fn, eps=1e-3, reasoning_iters=10, verbose=False, debug=False, ret_vals=False):
        assert 0 <= reasoning_iters <= max_iters
        v_delta_max = None
        if verbose: print("Learning values [ ", end="", flush=True)
        while self.iterno < max_iters - reasoning_iters:
            if verbose and (self.iterno % 30 == 0 or self.iterno == max_iters - reasoning_iters - 1):
                print(" {}".format(self.iterno), end="" if self.iterno == 0 or self.iterno % 300 else "\n\t",
                      flush=True)
            v_delta_max = self.step(policy_fn, debug=debug, ret_vals=False)
            if v_delta_max <= eps:
                break
        if self.iterno == max_iters - reasoning_iters:
            if verbose: print(" ] Stopped @ {}.".format(self.iterno))
            converged = False
        else:
            if verbose: print(" ] Converged @ {}.".format(self.iterno))
            converged = True

        if verbose: print("Reasoning [ ", end="", flush=True)
        stopped_at = self.iterno
        self.start_reasoning = True
        k = 0
        v_delta_max = None
        while self.iterno - stopped_at < reasoning_iters:
            if verbose and ((self.iterno - stopped_at) % 30 == 0 or self.iterno - stopped_at == reasoning_iters - 1):
                print(" {}".format(self.iterno),
                      end="" if self.iterno == stopped_at or self.iterno - stopped_at % 300 else "\n\t", flush=True)
            v_delta_max = self.step(policy_fn, debug=debug, ret_vals=False)
            if v_delta_max <= eps:
                converged = True
            k += 1

        if verbose: print(" ] Done ({} iters).".format(reasoning_iters))
        self.start_reasoning = False
        self.v_delta_max = v_delta_max
        self.converged = converged
        if ret_vals:
            return self.Pi, self.V, self.Q, self.iterno, v_delta_max, converged
        else:
            return


# heuristic_l2 = lambda s1, s2: np.linalg.norm(np.array(s1) - np.array(s2))
#
#
# def value_iteration(S, A, R, T, policy, gamma, n_iters, start=None, goal=None, step_cost=0.,convergence_eps=1e-3,
#                     verbose=False, dtype=torch.float32):
#
#     # assert torch.all(R < 0)
#     nS, nA = len(S), len(A)
#     v_delta_max = float("inf")
#     s_to_idx = {v:k for k,v in enumerate(S)}
#     a_to_idx = {a:i for i,a in enumerate(A)}
#
#     Q = torch.zeros(nS, nA, dtype=dtype)
#     log_Pi = torch.log(torch.ones(nS, nA, dtype=dtype) / nA)
#     # R can be list of differentiable functions, or a differentiable vector.
#     # Intialize V. (Can't make differentiable because it'll be overwritten)
#     V = torch.tensor([r for r in R], requires_grad=False)
#
#     # Given start
#     if start:
#         V_min = 10 *R.min() * 1 / (1-gamma)
#         start_state_idx = s_to_idx[start]
#         V[start_state_idx] = torch.tensor(V_min)
#     # Given goal
#     if goal:
#         V_max = torch.tensor(0)
#         goal_state_idx = s_to_idx[goal]
#         V[goal_state_idx] = torch.tensor(0)
#
#     if verbose: print("Running VI [ ", end="")
#     iterno = 0
#     while iterno < n_iters and v_delta_max > convergence_eps:
#
#         if verbose and iterno and iterno % 30 == 0:
#             print(".", end="" if iterno % 300 else "\n\t")
#
#         v_delta_max = 0
#         for si, s in enumerate(S):
#
#             v_s_prev = V[si].detach().item()
#             if si == goal_state_idx or s.is_terminal():
#                 continue
#
#             for ai, a in enumerate(A):
#
#                 for s_prime, p in T(s,a):
#
#                     if s_prime is None: # outside envelope
#                         continue
#                     Q[si, ai] += -step_cost + R[si] + gamma * p * V[s_to_idx[s_prime]].clone()
#
#             # Softmax action selection
#             log_Pi[si, :] = policy(Q[si, :].clone())
#             # Softmax value
#             V[si] = torch.exp(log_Pi[si, :].clone()).dot(Q[si, :].clone())
#             v_delta_max = max(abs(v_s_prev - V[si].detach().item()), v_delta_max)
#
#         iterno += 1
#
#     if iterno == n_iters:
#         if verbose: print(" ] VI didn't converge by {}.".format(iterno))
#     else:
#         if verbose: print(" ] VI converged @ {}.".format(iterno))
#
#     return log_Pi, V, Q, s_to_idx, a_to_idx, iterno
#
# def astar_find_path(start, goal, actions, trans_func, cost_fn=lambda s: 1, heuristic_fn=heuristic_l2):
#
#     frontier = PriorityQueue.PriorityQueue()
#     frontier.append([cost_fn(start)+heuristic_fn(start, goal), cost_fn(start), start])
#     explored = defaultdict(lambda: False)
#     cost = defaultdict(lambda: np.float("inf"))
#     parent = defaultdict(lambda: None)
#     path = []
#
#     while frontier.size():
#
#         f, g, s = frontier.pop()
#         explored[s] = True
#
#         if s == goal:
#
#             curr_state = goal
#             s_list = [curr_state]
#             a_list = [None]
#             parent_state = parent[curr_state] # greedy selection
#
#             while parent_state is not None:
#
#                 s_list.append(parent_state)
#                 a_list.append(get_action_from_state_change(parent_state, curr_state))
#
#                 curr_state = parent_state
#                 parent_state = parent[curr_state] # greedy selection
#
#             return list(zip(s_list[::-1], a_list[::-1]))
#
#         for a in actions:
#
#             sp = trans_func(s, a)
#             if not explored[sp] and sp not in frontier:
#
#                 g_new = g + cost_fn(sp)
#                 f_new = g_new + heuristic_fn(sp, goal)
#
#                 # prevent cycles
#                 if g_new < cost[sp]:
#                     frontier.append((f_new, g_new, sp))
#                     parent[sp] = s
#                     cost[sp] = g_new
#     return None
#
#
# def get_action_from_state_change(state, n_state):
#
#     dx = n_state[0] - state[0]
#     dy = n_state[1] - state[1]
#     if dx > 0 and dy == 0: return "right"
#     elif dx > 0 and dy > 0: return "up-right"
#     elif dx == 0 and dy > 0: return "up"
#     elif dx < 0 and dy > 0: return "up-left"
#     elif dx < 0 and dy == 0: return "left"
#     elif dx < 0 and dy < 0: return "down-left"
#     elif dx == 0 and dy < 0: return "down"
#     elif dx > 0 and dy < 0: return "down-right"
#     else: return "stay"
