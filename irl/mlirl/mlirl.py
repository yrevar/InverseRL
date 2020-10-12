import time, torch, numpy as np
from collections import defaultdict
import rl.planning as Plan
import rl.policy as Policy

from matplotlib import cm as cm, pyplot as plt, colors as mplotcolors


def run_mlirl(tau_lst, S, PHI, T, R_model, gamma=0.95, mlirl_iters=100,
              vi_max_iters=150, reasoning_iters=50, policy=lambda q: Policy.Boltzmann(q, boltzmann_temp=0.01),
              vi_eps=1e-4, checkpoint_freq=10, store_dir="./data/mlirl"):

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
            for tau in tau_lst:
                goal = tau[-1][0]
                if VI_by_goal[goal] is None:
                    # Run VI for this goal
                    print("Running VI (goal: {})".format(goal))
                    VI_by_goal[goal] = Plan.ValueIteration(S, R_curr, T, verbose=True, log_pi=False, gamma=gamma, goal=goal)
                    VI_by_goal[goal].run(vi_max_iters, policy, reasoning_iters=reasoning_iters, verbose=True, debug=False, eps=vi_eps)
                for s, a in tau:
                    if a is not None: # terminal state action is assumed None
                        # loss += -torch.log(VI_by_goal[goal].Pi[VI_by_goal[goal].get_tbl_idxs(S.at_loc(s), a)])
                        # laplace smoothing
                        loss += -torch.log(VI_by_goal[goal].Pi[VI_by_goal[goal].get_tbl_idxs(S.at_loc(s), a)] + 1e-20)

            ll = np.exp(-loss.detach().item())
            log_likelihoods_history.append(ll)
            VI_by_goal_bkp = VI_by_goal.copy()
            print(">>> Iter: {:04d} ({:03.3f}s): loss = {:09.6f}, likelihood = {:02.4f}\n\n".format(
                _iter, time.time( ) -_iter_start_time, loss, ll))

            if _iter % checkpoint_freq == 0 or _iter == mlirl_iters - 1:
                R_model.save(store_dir=store_dir, fname="mlirl_state_iter_{}.pth".format(_iter))

            loss.backward()
            bottleneck_grads = R_model.bottleneck_grads()
            bottleneck_grad_history.append(bottleneck_grads.numpy().copy())
            r_history.append(R_curr.detach().numpy().squeeze())
            print("Reward: ", R_curr.detach().numpy().squeeze())
            print("Bottleneck grads: ", bottleneck_grads)
            plt.imshow(bottleneck_grads)
            plt.title(str(bottleneck_grads))
            R_model.step()

    except KeyboardInterrupt:
        print("\nTraining interrupted @ iter {}".format(_iter))
        pass
    return log_likelihoods_history, R_curr, VI_by_goal_bkp, bottleneck_grad_history, r_history
