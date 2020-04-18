import time, torch, numpy as np
from collections import defaultdict
import rl.planning as Plan
import rl.policy as Policy

def run_mlirl(tau_lst, S, PHI, T, R_model, gamma=0.95, mlirl_iters=100,
                     vi_max_iters=150, reasoning_iters=50, boltzmann_temp=0.01, vi_eps=1e-4):

    # assert len(set([tau[-1][0] for tau in tau_lst])) <= 1, "All trajectories must have same goal."

    log_likelihoods_history = []
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
                    VI_by_goal[goal].run(vi_max_iters,
                                         lambda q: Policy.Boltzmann(q, boltzmann_temp=boltzmann_temp),
                                         reasoning_iters=reasoning_iters, verbose=True, debug=False, eps=vi_eps)
                for s, a in tau:
                    if a is not None: # terminal state action is assumed None
                        loss += -torch.log(VI_by_goal[goal].Pi[VI_by_goal[goal].get_tbl_idxs(S.at_loc(s), a)])

            ll = np.exp(-loss.detach().item())
            log_likelihoods_history.append(ll)
            VI_by_goal_bkp = VI_by_goal.copy()
            print(">>> Iter: {:04d} ({:03.3f}s): loss = {:09.6f}, likelihood = {:02.4f}\n\n".format(
                _iter, time.time( ) -_iter_start_time, loss, ll))
            loss.backward()
            R_model.step()

    except KeyboardInterrupt:
        print("\nTraining interrupted @ iter {}".format(_iter))
        pass
    return log_likelihoods_history, R_curr, VI_by_goal_bkp
