# Visualization functions 
import io
import numpy as np
import imageio
from PIL import Image
from IPython import display
import matplotlib.pyplot as plt

def read_pil_image_from_plt(plt):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)

def create_gif(img_generator, cmap=plt.cm.viridis, gif_name="./__gif_sample.gif", fps=10,
               figsize=(4, 4), title=None, display=False):
    with imageio.get_writer(gif_name, mode='I', fps=fps) as writer:
        for img in img_generator():
            # Append to GIF
            writer.append_data(np.array(img))
            
            # Wait to draw - only for online visualization
            if display:
                plt.imshow(img)
                display.clear_output(wait=True)
                display.display(plt.gcf())
        plt.clf()
    return

def convert_to_grid(nvmdp, S, state_values):
    
    grid = np.zeros((nvmdp.height, nvmdp.width))
    for si, s in enumerate(S):
        x, y = s
        row, col = nvmdp._xy_to_rowcol(x, y)
        grid[row, col] = state_values[si]
    return grid

def get_grad_evolution_images(nvmdp, S, R_grid, expert_traj, R_grads_evolution, boltzmann_temp, 
                              figsize=(40,20), R_kind=""):

    for i in range(len(R_grads_evolution)):
        
        plt.close('all')
        plt.figure(figsize=figsize)
        nvmdp.visualize_grid(R_grid, trajectories=expert_traj, cmap=plt.cm.Reds_r,
                             state_space_cmap=False, show_colorbar=True, fig=fig, subplot_str="121",
                             title="Navigation IRL MDP(with true reward). Expert Path (black).", end_marker="*c")
        nvmdp.visualize_grid(convert_to_grid(nvmdp, S, R_grads_evolution[i]), trajectories=[expert_traj[0][:i+1]], 
                             state_space_cmap=False,  traj_marker='-b',
                             cmap=plt.cm.Reds_r, show_colorbar=True, fig=fig, subplot_str="122",
                             title="MLIRL Gradients. {}, temp: {}, step: {}.".format(R_kind, boltzmann_temp, i+1))
        yield read_pil_image_from_plt(plt)
        plt.clf()

def plot_irl_gridworld(nvmdp, phi, state_ids, fig, traj_states, subplot_str="32", plot_idx=1):
    
    if fig is None:
        fig = plt.figure(figsize=(20,10))
        
    cdict = {0: "lightgrey", 1: "red"}
    group = state_ids
    labels = {0: "road", 1: "obstacle"}
    ax = fig.add_subplot(subplot_str + str(plot_idx))
    fig, ax = nvmdp.visualize_grid(trajectories=traj_states, 
                         state_space_cmap=True,
                         show_rewards_colorbar = True,
                         cmap=plt.cm.Reds_r, 
                         show_colorbar=True, 
                         init_marker=".",
                         end_marker="*", 
                         traj_linewidth=2,
                         fig=fig, ax=ax, plot=False)

    ax = fig.add_subplot(subplot_str + str(plot_idx+1))
    for g in np.unique(group):
        ix = np.where(group == g)
        ax.scatter(phi[ix,0], phi[ix,g], c = cdict[g>0], label = labels[g>0], s = 50)
        # ax.scatter(phi[ix,0], phi[ix,1], c = cdict[g], label = labels[g], s = 50)
    # plt.scatter(phi[:,0], phi[:,1], c=[colors[s] for s in state_ids], g=state_ids, label=["obstacle", "drivable"])
    plt.tight_layout()
    plt.title("Features")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
#     plt.legend()
    return fig, ax

def plot_irl_results(nvmdp, fig, R_rec, loss_history, traj_states, is_loglik_loss=True,
                     title="MLIRL (specifics)", subplot_str="32", plot_idx=3, ylabel=""):
    
    if fig is None:
        fig = plt.figure(figsize=(20,10))
        
    ax = fig.add_subplot(subplot_str + str(plot_idx))
    fig, ax = nvmdp.visualize_grid(R_rec, trajectories=traj_states, 
                         state_space_cmap=False,
                         show_rewards_colorbar=False,
                         cmap=plt.cm.gray, 
                         show_colorbar=True, 
                         init_marker=".",
                         end_marker="*", 
                         traj_linewidth=2,
                         fig=fig, ax=ax, plot=False, title=title + ": Recovered Reward")

    ax = fig.add_subplot(subplot_str + str(plot_idx+1))
    
    if is_loglik_loss:
        plt.plot(np.exp(-np.asarray(loss_history)))
        plt.ylabel("Likelihood")
    else:
        plt.plot(loss_history)
        if ylabel:
            plt.ylabel(ylabel)
        else:
            plt.ylabel("Loss")
        
    plt.xlabel("Iterations")
    plt.title(title + ": Training Curve")
    
    return fig, ax

def state_id_to_linear_feature(state_id):
    
    sigma = 0.1
    
    if state_id == 0: # drivable 
        if np.random.random() > 0.5:
            mu1 = -0.5
            mu2 = -0.5
        else:
            mu1 = -0.5
            mu2 = 0.5
        return [np.random.normal(mu1, sigma), np.random.normal(mu2, sigma)]
    else: # obstacle
        if np.random.random() > 0.5:
            mu1 = 0.5
            mu2 = -0.5
        else:
            mu1 = 0.5
            mu2 = 0.5
        return [np.random.normal(mu1, sigma), np.random.normal(mu2, sigma)]
    

def state_id_to_non_linear_feature(state_id):
    
    sigma = 0.1
    
    if state_id == 0: # drivable 
        if np.random.random() > 0.5:
            mu1 = -0.5
            mu2 = -0.5
        else:
            mu1 = 0.5
            mu2 = 0.5
        return [np.random.normal(mu1, sigma), np.random.normal(mu2, sigma)]
    else: # obstacle
        if np.random.random() > 0.5:
            mu1 = -0.5
            mu2 = 0.5
        else:
            mu1 = 0.5
            mu2 = -0.5
        return [np.random.normal(mu1, sigma), np.random.normal(mu2, sigma)]
    
    