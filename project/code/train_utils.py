import torch
import numpy as np
import utils

def get_encoding(encoder, ego_states, stl_embeds, stl_str, args):
    if args.zero_ego:
        ego_states = ego_states * 0
    conditions = encoder(ego_states, stl_embeds)
    if args.pretraining:
        conditions = conditions * 0
    return conditions

def normalize_traj(origin_traj, stat_mean, stat_std):
    assert len(origin_traj.shape) == 3
    assert len(stat_mean.shape) == len(stat_std.shape)
    assert origin_traj.shape[-1] == stat_mean.shape[-1] and origin_traj.shape[-1] == stat_std.shape[-1]
    norm_traj = (origin_traj - stat_mean) / (stat_std.clip(min=1e-4))
    return norm_traj

def denorm_traj(norm_traj, stat_mean, stat_std):
    assert len(norm_traj.shape) == 3
    assert len(stat_mean.shape) == len(stat_std.shape)
    assert norm_traj.shape[-1] == stat_mean.shape[-1] and norm_traj.shape[-1] == stat_std.shape[-1]
    origin_traj = norm_traj * stat_std + stat_mean
    return origin_traj

def parse_batch(batch, device):
    index_tensors, stl_embeds, ego_states, us, sa_trajectories, stl_str, sa_par, stl_i_tensors, stl_type_i_tensors = batch
    index_tensors = index_tensors.to(device)
    stl_embeds = stl_embeds.to(device)
    ego_states = ego_states.to(device)
    us = us.to(device)
    sa_trajectories = sa_trajectories.to(device)
    stl_str = stl_str.to(device)
    sa_par = sa_par.to(device)
    stl_i_tensors = stl_i_tensors.to(device)
    stl_type_i_tensors = stl_type_i_tensors.to(device)
    return index_tensors, stl_embeds, ego_states, us, sa_trajectories, stl_str, sa_par, stl_i_tensors, stl_type_i_tensors

def get_denoising_results(diffuser, conditions, stat_mean, stat_std, args, guidance_data=None):
    results = diffuser.conditional_sample(conditions, horizon=None, in_painting=None, guidance_data=guidance_data, args=args)
    diffused_trajs = denorm_traj(results.trajectories, stat_mean=stat_mean, stat_std=stat_std)[:, :, :2]
    return diffused_trajs

def set_model(net, mode):
    if mode == "train":
        net.train()
    else:
        net.eval()

def mean_func(x):
    if len(x) == 0:
        return 0
    else:
        return np.mean([utils.to_np(xx) for xx in x])