import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import utils
import generate_scene_v1
import numpy as np
import wandb

def visualize_plan(mode, bi, mini_i, gt_trajs_np, diffused_trajs_np, 
                   simple_stl, objects_d, args, 
                   res_trajs=None, x_lims=(-5,5), y_lims=(-5,5), wandb_log=True, step=None):
    
    fig = plt.figure(figsize=(12, 4) if res_trajs is not None else (8, 8))
    
    # Plot Trajectories
    plt.subplot(1, 2, 1) if res_trajs is not None else plt.subplot(1, 1, 1)
    ax = plt.gca()
    for obj_i, obj_keyid in enumerate(objects_d):
        obj_d = objects_d[obj_keyid]
        # Handle 'is_obstacle' check safely
        color = "royalblue"
        if "is_obstacle" in obj_d and obj_d["is_obstacle"]:
            color = "gray"
        elif obj_d["r"] < 0.75: # Fallback heuristic from original code
             color = "royalblue"
        else:
             color = "gray"
            
        ax.add_patch(Circle([obj_d["x"], obj_d["y"]], radius=obj_d["r"], color=color, alpha=0.5))
        plt.text(obj_d["x"], obj_d["y"], s="%d" % (obj_keyid))

    plt.scatter(gt_trajs_np[0, 0], gt_trajs_np[0, 1], color="purple", s=64, zorder=1000)
    plt.plot(gt_trajs_np[:, 0], gt_trajs_np[:, 1], color="green", alpha=0.3, linewidth=4, zorder=999)

    # Plot diffused samples
    plt.scatter(diffused_trajs_np[:, 0, 0], diffused_trajs_np[:, 0, 1], color="orange", alpha=0.05, s=48, zorder=1000)
    for i in range(diffused_trajs_np.shape[0]):
        plt.plot(diffused_trajs_np[i, :, 0], diffused_trajs_np[i, :, 1], linewidth=2, color="brown", alpha=0.3)

    if res_trajs is not None:
        plt.scatter(res_trajs[0, 0], res_trajs[0, 1], color="orange", alpha=0.5, s=48, zorder=20)
        plt.plot(res_trajs[:, 0], res_trajs[:, 1], linewidth=2, color="brown", alpha=0.8, zorder=1500)

    plt.axis("scaled")
    plt.xlim(x_lims)
    plt.ylim(y_lims)

    # Plot Tree (only if doing full eval viz)
    if res_trajs is not None:
        plt.subplot(1, 2, 2)
        generate_scene_v1.plot_tree(simple_stl)
        plt.xticks([])
        plt.yticks([])
    
    if wandb_log and not args.dryrun:
        key = f"{mode}_viz/b{bi:04d}_{mini_i}" if step is None else f"{mode}_viz/epi{step:04d}_b{bi}"
        wandb.log({key: wandb.Image(plt)})
    
    utils.plt_save_close("%s/viz_%s_b%04d_%d.png" % (args.viz_dir, mode, bi, mini_i))