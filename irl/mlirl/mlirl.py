import time, yaml, torch, numpy as np, os.path as osp
from collections import defaultdict
import rl.planning as Plan
import rl.policy as Policy
import utils
# from matplotlib import cm as cm, pyplot as plt, colors as mplotcolors

class MLIRL(object):

    def __init__(self, store_dir="./data/mlirl", verbose=False, debug=False):
        self.store_dir = store_dir
        self.verbose = verbose
        self.debug = debug
        self.state_file = osp.join(store_dir, "mlirl_state.yaml")
        self.reset()

    def set(self, likelihoods_history, converged_history, bottleneck_grad_history, iter_trained, reward_model_ckpt):
        self.likelihoods_history = likelihoods_history
        self.converged_history = converged_history
        # self.reward_history = []
        self.bottleneck_grad_history = bottleneck_grad_history
        self.iter_trained = iter_trained
        self.reward_model_ckpt = reward_model_ckpt

    def reset(self):
        # saw weird behavior with mutable argument initialization, hence separate reset() method
        self.likelihoods_history = []
        self.converged_history = []
        # self.reward_history = []
        self.bottleneck_grad_history = []
        self.iter_trained = 0
        self.reward_model_ckpt = None

    def resume(self):
        if osp.exists(self.state_file):
            self.load_state(self.state_file)

    def save_state(self, file_name):
        state = {
            "likelihoods_history": self.likelihoods_history,
            "converged_history": self.converged_history,
            "bottleneck_grad_history": self.bottleneck_grad_history,
            "iter_trained": self.iter_trained,
            "reward_model_ckpt": self.reward_model_ckpt
        }
        with open(file_name, 'w') as f:
            yaml.dump(state, f)

    def load_state(self, file_name):
        with open(file_name, 'r') as f:
            state = yaml.load(f, Loader=yaml.Loader)
        self.set(**state)

    def train(self, grid_world_list, reward_spec, n_iters, policy, vi_max_iters=10, reasoning_iters=5,
              vi_eps=1e-6, gamma=0.95, checkpoint_freq=1e-10, cost_reg_lambda=0.0):
        _iter = 0
        reward_model = reward_spec.get_model()
        try:
            for _iter in range(self.iter_trained, self.iter_trained + n_iters):
                _iter_start_time = time.time()
                converged_status_list = []
                loss = 0
                reward_model.zero_grad()
                for grid_world in grid_world_list:
                    for goal, traj_list in grid_world.get_trajectories().read_by_goals(zip_sa=True):
                        vi = grid_world.plan(goal,
                                             policy=policy,
                                             vi_max_iters=vi_max_iters,
                                             reasoning_iters=reasoning_iters,
                                             vi_eps=vi_eps,
                                             gamma=gamma,
                                             reward_key=reward_spec.get_key(),
                                             verbose=self.verbose,
                                             debug=self.debug)
                        converged_status_list.append(vi.converged)
                        if cost_reg_lambda == 0.0:
                            loss -= utils.log_likelihood(vi.Pi, traj_list)
                        else:
                            loss -= utils.log_likelihood(vi.Pi, traj_list) + cost_reg_lambda * utils.cost_regularization_term(
                                r_loc_fn=lambda s: grid_world.VI[goal].R[
                                    grid_world.VI[goal].s_to_idx[grid_world.S.at_loc(s)]],
                                traj_list=traj_list
                            )

                ll = np.exp(-loss.detach().item())
                all_converged =  np.all(converged_status_list)
                self.converged_history.append(all_converged)
                self.likelihoods_history.append(ll)

                print(">>> Iter: {:04d} ({:03.3f}s): VI converged {}, loss {:09.6f}, likelihood {:02.4f}".format(
                    self.iter_trained, time.time() - _iter_start_time, all_converged, loss, ll), end="\n" if self.verbose else "\r")

                if self.iter_trained % checkpoint_freq == 0 or self.iter_trained == n_iters - 1:
                    reward_model.save(store_dir=self.store_dir, fname="mlirl_reward_state_iter_{}.pth".format(self.iter_trained))
                    self.reward_model_ckpt = self.iter_trained
                loss.backward()

                bottleneck_grads = reward_model.bottleneck_grads()
                self.bottleneck_grad_history.append(bottleneck_grads.numpy().copy())
                # self.reward_history.append(reward_model.detach().numpy().squeeze())
                reward_model.step()
                self.iter_trained += 1
                self.save_state(self.state_file)
        except KeyboardInterrupt:
            print("\nTraining interrupted @ iter {}".format(_iter))
            reward_model.save(store_dir=self.store_dir, fname="mlirl_reward_state_iter_{}.pth".format(_iter))
            self.reward_model_ckpt = self.iter_trained
            self.save_state(self.state_file)
        return


def run_mlirl(tau_lst, S, PHI, T, R_model, gamma=0.95, mlirl_iters=100,
              vi_max_iters=150, reasoning_iters=50, policy=lambda q: Policy.Boltzmann(q, boltzmann_temp=0.01),
              vi_eps=1e-4, checkpoint_freq=1e10, store_dir="./data/mlirl", verbose=False, debug=False):

    # assert len(set([tau[-1][0] for tau in tau_lst])) <= 1, "All trajectories must have same goal."

    log_likelihoods_history = []
    bottleneck_grad_history = []
    r_history = []
    VI_by_goal_bkp = None
    try:
        for _iter in range(mlirl_iters):

            _iter_start_time = time.time()
            VI_by_goal = defaultdict(lambda: None)
            R_model.zero_grad()
            R_curr = R_model(PHI, return_latent=False)

            loss = 0
            converged_status_list = []
            for tau in tau_lst:
                goal = tau[-1][0]
                if VI_by_goal[goal] is None:
                    # Run VI for this goal
                    if verbose: print("Running VI (goal: {})".format(goal))
                    VI_by_goal[goal] = Plan.ValueIteration(S, R_curr, T, verbose=verbose,
                                                           log_pi=False, gamma=gamma, goal=goal)
                    _Pi, _V, _Q, _iterno, _v_delta_max, _converged = \
                        VI_by_goal[goal].run(vi_max_iters, policy, reasoning_iters=reasoning_iters,
                                         verbose=verbose, debug=debug, eps=vi_eps, ret_vals=True)
                    converged_status_list.append(_converged)
                for s, a in tau:
                    if a is not None: # terminal state action is assumed None
                        # loss += -torch.log(VI_by_goal[goal].Pi[VI_by_goal[goal].get_tbl_idxs(S.at_loc(s), a)])
                        # laplace smoothing
                        loss += -torch.log(VI_by_goal[goal].Pi[VI_by_goal[goal].get_tbl_idxs(S.at_loc(s), a)] + 1e-20)

            ll = np.exp(-loss.detach().item())
            log_likelihoods_history.append(ll)
            VI_by_goal_bkp = VI_by_goal.copy()
            print(">>> Iter: {:04d} ({:03.3f}s): VI converged {}, loss {:09.6f}, likelihood {:02.4f}\r".format(
                _iter, time.time( ) -_iter_start_time, np.all(converged_status_list), loss, ll), end="")

            if _iter % checkpoint_freq == 0 or _iter == mlirl_iters - 1:
                R_model.save(store_dir=store_dir, fname="mlirl_state_iter_{}.pth".format(_iter))

            loss.backward()
            bottleneck_grads = R_model.bottleneck_grads()
            bottleneck_grad_history.append(bottleneck_grads.numpy().copy())
            r_history.append(R_curr.detach().numpy().squeeze())
            if debug: print("Reward: ", R_curr.detach().numpy().squeeze())
            if debug: print("Bottleneck grads: ", bottleneck_grads)
            R_model.step()

    except KeyboardInterrupt:
        print("\nTraining interrupted @ iter {}".format(_iter))
        R_model.save(store_dir=store_dir, fname="mlirl_state_iter_{}.pth".format(_iter))
        pass
    return log_likelihoods_history, R_curr, VI_by_goal_bkp, bottleneck_grad_history, r_history
