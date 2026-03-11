import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import utils
import tqdm

from z_diffuser import GaussianDiffusion, GaussianFlow, TemporalUnet, MockNet, MLPNet, GaussianVAE, GradNN
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter
from torch_geometric.data import Dataset  #, DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.transforms import two_hop
import generate_scene_v1

from matplotlib.patches import Polygon, Rectangle, Ellipse, Circle
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from generate_scene_v1 import SimpleUntil, SimpleF, SimpleG, SimpleReach

from stl_to_seq_utils import stl_to_seq, rand_aug, hard_rand_aug, aug_graph, compute_tree_size, stl_hash_seq, stl_naive_str, stl_smart_encode, stl_to_1d_array
        
from z_models import GCN, ScorePredictor, MLP

from os.path import join as ospj


def load_dataset():
    data_name = "data_mini.npz"
    stl_data_all = np.load(ospj(utils.get_exp_dir(), args.data_path, data_name), allow_pickle=True)['data']
    if not args.concise:
        print("Finished loading numpy data")
        print("Make absolute idx...")
    for rec_i, record in enumerate(tqdm.tqdm(stl_data_all)):
        record["abs_idx"] = rec_i
    if not args.concise:
        print("Finished marking idx")
    
    if args.select_indices is not None:
        stl_data_all = [stl_data_all[idx] for idx in args.select_indices]
    if args.clip_max:
        original_len = len(stl_data_all)
        stl_data_all = stl_data_all[:args.clip_max]
        if not args.concise:
            print("Clip the dataset from %d to %d => %d"%(original_len, args.clip_max, len(stl_data_all)))
    
    type_stat={}
    type_init_stat={}
    type_sol_stat={}
    demo_list = stl_data_all
    total_nums = len(demo_list)
    
    # do the flattening
    simple_stl_list = []
    real_stl_list = []
    stl_data_list = []
    obj_list = []
    stl_str_list = []
    type_list = []
    cache = {}
    
    rec_i_2_stl_i = {}
    stl_i_2_rec_i = {}
    
    curr_add = []
    origin_tree_size=[]
    curr_tree_size = []
    if not args.concise:
        print("Load STLs ...")
    visited_stl_ids = []
    for rec_i, record in enumerate(tqdm.tqdm(stl_data_all)):
        stl_key_id = rec_i

        if stl_key_id not in visited_stl_ids:
            visited_stl_ids.append(stl_key_id)

            obj_d = {}
            simple_stl_list.append(
                generate_scene_v1.find_ap_in_lines(0, stl_dict={}, objects_d=obj_d, lines=record["stl"], numpy=True, real_stl=False, until1=False))
            obj_list.append(dict(obj_d))
            
            if args.aug_graph:
                cnt_d={"n":1}
                compute_tree_size(simple_stl_list[-1], cnt_d=cnt_d)
                origin_tree_size.append(cnt_d["n"])
                curr_cnt_stat = {'add':0}
                simple_stl_list[-1] = aug_graph(simple_stl_list[-1], cfg={"tmax":args.horizon}, inplace=False, max_aug=args.max_aug, curr_cnt_stat=curr_cnt_stat)
                curr_add.append(curr_cnt_stat["add"])
                cnt_d={"n":1}
                compute_tree_size(simple_stl_list[-1], cnt_d=cnt_d)
                curr_tree_size.append(cnt_d["n"])
                
            else:
                cnt_d={"n":1}
                compute_tree_size(simple_stl_list[-1], cnt_d=cnt_d)
                curr_add.append(0)
                curr_tree_size.append(cnt_d["n"])
                origin_tree_size.append(cnt_d["n"])
            
            real_stl_list.append(
                generate_scene_v1.find_ap_in_lines(0, stl_dict={}, objects_d={}, lines=record["stl"], numpy=True, real_stl=True,
                                                   ap_mode="l2", until1=False))
            type_list.append(record['stl_type_i'])
            stl_data_list.append(record["stl"])
            stl_str_list.append([0])

        rec_i_2_stl_i[rec_i] = len(visited_stl_ids)-1
        stl_i_2_rec_i[len(visited_stl_ids)-1] = rec_i
    
    if not args.concise:
        print("TREESIZE original:%.3f add:%.3f new:%.3f"%(np.mean(origin_tree_size), np.mean(curr_add),  np.mean(curr_tree_size)))
        # filter out non-sat cases
        # abs_stl_index, abs_ego_i, abs_sol_i, rel_stl_index, rel_ego_i, rel_sol_i
        print("Filter data...")
    file_list = []
    rel_stl_index=0
    
    num_egos = 8
    num_inits = 8

    if args.type_ratios is not None:
        assert len(args.type_ratios)==4
        ratios = np.array(args.type_ratios)
        ratios = ratios/np.sum(ratios)
        max_ratio = np.max(ratios)
        current_stl_cnt = {0:0, 1:0, 2:0, 3:0}
        expect_stl_quota = {0:0, 1:0, 2:0, 3:0}
        if args.max_sol_clip is not None:
            allowed_sol_each = args.max_sol_clip
        else:
            if args.first_sat_init:
                allowed_sol_each = 1 * num_inits
            else:
                allowed_sol_each = num_egos * num_inits
        
        unique, counts = np.unique(type_list, return_counts=True)
        raw_stl_ests = dict(zip(unique, counts))
        S = np.min([
            len(visited_stl_ids), 
            raw_stl_ests[0] / ratios[0] if 0 in raw_stl_ests else len(visited_stl_ids),
            raw_stl_ests[1] / ratios[1] if 1 in raw_stl_ests else len(visited_stl_ids),
            raw_stl_ests[2] / ratios[2] if 2 in raw_stl_ests else len(visited_stl_ids),
            raw_stl_ests[3] / ratios[3] if 3 in raw_stl_ests else len(visited_stl_ids),
        ])
        for type_ii in range(4):
            expect_stl_quota[type_ii] = (S * allowed_sol_each) * ratios[type_ii]
        
        print(expect_stl_quota)
    
    for rec_i, record in enumerate(tqdm.tqdm(demo_list)):
        score = record['score'].reshape(num_egos, num_inits)
        if "trajs" not in record:
            ego_state = torch.from_numpy(record['state']).float()
            us = torch.from_numpy(record['us']).float().reshape(-1, record['us'].shape[-2], record['us'].shape[-1])
            record["trajs"] = utils.to_np(generate_scene_v1.generate_trajectories(ego_state, us, args.dt))

        stl_type_i = record['stl_type_i']

        if args.filtered_types is not None and stl_type_i not in args.filtered_types:
            continue

        if stl_type_i not in type_stat:
            type_stat[stl_type_i] = 0
            type_init_stat[stl_type_i] = 0
            type_sol_stat[stl_type_i] = 0

        rel_ego_i=0
        for ego_i in range(num_egos):
            rel_sol_i=0
            for sol_i in range(num_inits):
                if score[ego_i, sol_i]>0:
                    if args.type_ratios is not None:
                        if current_stl_cnt[stl_type_i] == expect_stl_quota[stl_type_i]:
                            continue
                        current_stl_cnt[stl_type_i]+=1

                    file_list.append([rec_i, rec_i, ego_i, sol_i, rel_stl_index, rel_ego_i, rel_sol_i])
                    rel_sol_i+=1

                if args.max_sol_clip is not None and rel_sol_i==args.max_sol_clip:
                    break

            if rel_sol_i>0:
                rel_ego_i+=1

            type_sol_stat[stl_type_i] += rel_sol_i
            if args.first_sat_init and rel_ego_i==1:
                break
        type_init_stat[stl_type_i] += rel_ego_i
        type_stat[stl_type_i] += (rel_ego_i>0)
        if rel_ego_i>0:
            rel_stl_index+=1
    if not args.concise:
        print("*"*10,"Stat","*"*10)
        if args.type_ratios is not None:
            print("Expect ratios: %.3f %.3f %.3f %.3f | Actual: %.3f %.3f %.3f %.3f"%(
                ratios[0], ratios[1], ratios[2], ratios[3],
                current_stl_cnt[0]/len(file_list), current_stl_cnt[1]/len(file_list),
                current_stl_cnt[2]/len(file_list), current_stl_cnt[3]/len(file_list),
            ))
        print("Sat cases:", type_stat)
        print("Sat inits:", type_init_stat)
        print("Sat sols:", type_sol_stat)
    if args.first_sat_init:
        print("LEN",total_nums,"SAT ratio",len(file_list)/(total_nums*num_inits))
    else:
        print("LEN",total_nums,"SAT ratio",len(file_list)/(total_nums*num_egos*num_inits))

    return demo_list, stl_data_list, simple_stl_list, real_stl_list, obj_list, file_list, stl_str_list, type_list, cache

# TODO(data)
class GSTLDataset(Dataset):
    def __init__(self, dataset, split, seq_max_len, embed_dim, shuffle=True):
        super().__init__(None, None, None, None)
        self.dataset = dataset
        self.split = split        
        _, _, _, _, _, file_list, _, _, _ = dataset        
        self.seq_max_len = seq_max_len
        self.embed_dim = embed_dim
        n_stls = file_list[-1][4]+1
        n_split = int(n_stls * 0.8)
        
        # for reproducibility across train/val split
        rng_state = torch.get_rng_state()
        perm_stl_indices = torch.randperm(n_stls, generator=torch.Generator().manual_seed(8008208820))
        torch.set_rng_state(rng_state)
        
        train_indices = perm_stl_indices[:n_split]
        val_indices = perm_stl_indices[n_split:]
        
        if split=="train":
            indices = train_indices
        elif split=="val":
            indices = val_indices
        elif split=="full":
            indices = list(range(n_stls))
        else:
            raise NotImplementedError
        
        self.fs_list = []
        for line in file_list:
            if line[4] in indices:
                self.fs_list.append(line)
        
        if shuffle:
            rng_state = torch.get_rng_state()
            perm_stl_indices = torch.randperm(len(self.fs_list), generator=torch.Generator().manual_seed(8008008888))
            self.fs_list = [self.fs_list[idxidx] for idxidx in perm_stl_indices]
            torch.set_rng_state(rng_state)

        # TODO change data indices during evaluation phase
        if args.test and args.seed != 1007:
            perm_stl_indices = torch.randperm(len(self.fs_list), generator=torch.Generator().manual_seed(args.seed))
            self.fs_list = [self.fs_list[idxidx] for idxidx in perm_stl_indices]
            
        if not args.concise:
            print("%s-len:%d"%(split, len(self.fs_list)))
        self.cache_tmp = None
        
    def len(self):
        return len(self.fs_list)
    
    def get(self, index):
        demo_list, stl_data_list, simple_stl_list, real_stl_list, obj_list, file_list, stl_str_list, type_list, cache = self.dataset
        rec_i, stl_i, ego_i, sol_i, rel_stl_index, rel_ego_i, rel_sol_i = self.fs_list[index]
        
        sa_partial = torch.tensor([0])
        
        if (args.rand_aug_eval==False and args.rand_aug_graph) or (args.rand_aug_eval and self.split=="val") or args.test:
            the_simple_stl = rand_aug(simple_stl_list[stl_i], inplace=False)
        else:
            the_simple_stl = simple_stl_list[stl_i]
        
        if args.data_aug and self.split=="train":
            the_simple_stl = hard_rand_aug(simple_stl_list[stl_i], cfg={"tmax":args.horizon}, inplace=False)
        
        
        record = demo_list[rec_i]
        
        ego_state = torch.from_numpy(record["state"][ego_i * 8 + sol_i]).float()
        us = torch.from_numpy(record["us"][ego_i, sol_i]).float()
        if "trajs" in record:
            traj = torch.from_numpy(record["trajs"][ego_i * 8 + sol_i]).float()
        else:
            traj = generate_scene_v1.generate_trajectories(ego_state, us, args.dt)
        sa_traj = torch.cat([traj[..., :-1, :], us], dim=-1)
        stl_embed = get_graph_stl_embed_from_tree(the_simple_stl, is_train=self.split=="train")
        index_tensor = torch.tensor(index)
        stl_i_tensor = torch.tensor(stl_i)
        stl_type_i_tensor = torch.tensor(record["stl_type_i"])
        stl_str = torch.ones(1)*-1
                    
        return index_tensor, stl_embed, ego_state, us, sa_traj, stl_str, sa_partial, stl_i_tensor, stl_type_i_tensor

HASHI_D = {
    2:   tuple([1, 0, 0, 0, 0, 0, 0,]),  #  negation
    0:   tuple([0, 1, 0, 0, 0, 0, 0,]),  #  conjunction
    1:   tuple([0, 0, 1, 0, 0, 0, 0,]),  #  disjunction
    5:   tuple([0, 0, 0, 1, 0, 0, 0,]),  #  eventually
    6:   tuple([0, 0, 0, 0, 1, 0, 0,]),  #  always
    7:   tuple([0, 0, 0, 0, 0, 1, 0,]),  #  until
    8:   tuple([0, 0, 0, 0, 0, 0, 0,]),  #  reach
}

def get_env_goals_embedding(obj_d_list):
    node_list = []
    edge_list = []
    # introduce one root node
    node_list.append([-1, 0, 0, 0, 0]) 
    for obj_i, obj_key in enumerate(obj_d_list):
        obj_d = obj_d_list[obj_key]
        node_list.append([obj_d["is_obstacle"]*1, obj_d['x'], obj_d['y'], obj_d['z'], obj_d['r']])
        edge_list.append([obj_i+1, 0])
    x = torch.Tensor(node_list).float()
    edge_index = torch.Tensor(edge_list).long().T
    graph_data = Data(x=x, edge_index=edge_index)
    return graph_data

def get_graph_stl_embed_from_tree(simple_stl, is_train=False):
    node_list = []
    edge_list = []
    depth_list = []
    total_i = 1
    queue = [(0, 0, simple_stl, -1, -1)]

    while len(queue) != 0:
        depth, ego_idx, node, left_child, right_child = queue[0]
        del queue[0]
        node_type = generate_scene_v1.check_stl_type(node)
        ta, tb = -1, -1
        obj_x, obj_y, obj_z, obj_r = -1, -1, -1, -1  
        if node_type != 8:
            ta = node.ts
            tb = node.te
            for _i in range(len(node.children)):
                if args.bidir:
                    edge_list.append([total_i, ego_idx])
                    edge_list.append([ego_idx, total_i])
                else:
                    edge_list.append([total_i, ego_idx])
                if node_type==7:
                    queue.append([depth+1, total_i, node.children[_i], 1 if _i==0 else -1, 1 if _i==1 else -1])
                else:
                    queue.append([depth+1, total_i, node.children[_i], -1, -1])
                total_i+=1
            
        else:
            obj_x = node.obj_x
            obj_y = node.obj_y
            obj_z = node.obj_z
            obj_r = node.obj_r
                    
        if args.normalize:
            x_scale = 5.0
            y_scale = 5.0
            node_feature = [node_type / 8, ta / args.horizon, tb / args.horizon, obj_x / x_scale, obj_y / y_scale, obj_z, obj_r, left_child]
        else:
            node_feature = [node_type, ta, tb, obj_x, obj_y, obj_z, obj_r, left_child]

        node_list.append(node_feature)
        depth_list.append(depth)
        
    x = torch.Tensor(node_list).float()
    depths = torch.Tensor(depth_list).long()
    edge_index = torch.Tensor(edge_list).long().T
    graph_data = Data(x=x, edge_index=edge_index, depths=depths)
    return graph_data


def get_data_loader(dataset, seq_max_len, embed_dim):
    if args.same_train_val:
        train_dataset = GSTLDataset(dataset, "full", seq_max_len, embed_dim)
        val_dataset = GSTLDataset(dataset, "full", seq_max_len, embed_dim)
    else:
        train_dataset = GSTLDataset(dataset, "train", seq_max_len, embed_dim)
        val_dataset = GSTLDataset(dataset, "val", seq_max_len, embed_dim)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=not(args.no_shuffle),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, val_loader


def get_encoding(encoder, ego_states, stl_embeds, stl_str):
    conditions = encoder(ego_states, stl_embeds)
    if args.pretraining:
        conditions = conditions * 0
    return conditions


def normalize_traj(origin_traj, mode="meanstd", stat_min=None, stat_max=None, stat_mean=None, stat_std=None, decay=0.9):
    assert len(origin_traj.shape)==3
    if mode=="meanstd": # (N, T, SA) (1, SA), Or (N, T, SA) (T, SA)
        assert len(stat_mean.shape)==len(stat_std.shape)
        assert origin_traj.shape[-1]==stat_mean.shape[-1] and origin_traj.shape[-1]==stat_std.shape[-1]
        norm_traj = (origin_traj - stat_mean) / (stat_std.clip(min=1e-4))
    elif mode=="minmax":
        norm_traj = (origin_traj - stat_min) / ((stat_max - stat_min).clip(min=1e-4))
        # (0, 1) -> (-decay, decay)
        norm_traj = (norm_traj - 0.5) * 2 * args.stat_decay
    else:
        raise NotImplementedError
    return norm_traj

def denorm_traj(norm_traj, mode="meanstd", stat_min=None, stat_max=None, stat_mean=None, stat_std=None, decay=0.9):
    assert len(norm_traj.shape)==3
    if mode=="meanstd": # (N, T, SA) (1, SA), Or (N, T, SA) (T, SA)
        assert len(stat_mean.shape)==len(stat_std.shape)
        assert norm_traj.shape[-1]==stat_mean.shape[-1] and norm_traj.shape[-1]==stat_std.shape[-1]
        origin_traj = norm_traj * stat_std + stat_mean
    elif mode=="minmax":
        # (-decay, decay) -> (0, 1)
        norm_traj = norm_traj / (2 * args.stat_decay) + 0.5
        origin_traj = norm_traj * (stat_max - stat_min) + stat_min
    else:
        raise NotImplementedError
    return origin_traj

def parse_batch(batch):
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

def get_denoising_results(diffuser, conditions, stat_mean, stat_std, stat_min, stat_max, par_indices, in_painting=None, guidance_data=None):
    if args.grad_nn:
        results = diffuser.forward(conditions)
        diffused_trajs = denorm_traj(results.trajectories, stat_mean=stat_mean, stat_std=stat_std)
        return diffused_trajs
    else:
        results = diffuser.conditional_sample(conditions, horizon=None, in_painting=in_painting, guidance_data=guidance_data, args=args)
        diffused_trajs = denorm_traj(results.trajectories, stat_mean=stat_mean, stat_std=stat_std)[:, :, :2]
        return diffused_trajs

def norm_func_decorator(args, mode, stat_min, stat_max, stat_mean, stat_std):
    def norm_func(origin_traj):
        assert len(origin_traj.shape)==3
        if mode=="meanstd": # (N, T, SA) (1, SA), Or (N, T, SA) (T, SA)
            assert len(stat_mean.shape)==len(stat_std.shape)
            assert origin_traj.shape[-1]==stat_mean.shape[-1] and origin_traj.shape[-1]==stat_std.shape[-1]
            norm_traj = (origin_traj - stat_mean) / (stat_std.clip(min=1e-4))
        elif mode=="minmax":
            norm_traj = (origin_traj - stat_min) / ((stat_max - stat_min).clip(min=1e-4))
            norm_traj = (norm_traj - 0.5) * 2 * args.stat_decay
        else:
            raise NotImplementedError
        return norm_traj
    return norm_func

def denorm_func_decorator(args, stat_min, stat_max, stat_mean, stat_std, par_indices):
    def denorm_func(results_trajectories):
        diffused_trajs = denorm_traj(results_trajectories, stat_mean=stat_mean, stat_std=stat_std)
        return diffused_trajs
    return denorm_func

def set_model(net, mode):
    if mode=="train":
        net.train()
    else:
        net.eval()

def sst_encoder(encoder, tuple_data, train_loader, val_loader):
    demo_list, stl_data_list, simple_stl_list, real_stl_list, obj_list, file_list, stl_str_list, type_list, cache = tuple_data    
    encoder = encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    eta = utils.EtaEstimator(start_iter=0, end_iter=args.epochs * len(train_loader), )
    scheduler = utils.create_custom_lr_scheduler(optimizer, warmup_epochs=args.warmup_epochs, warmup_lr=args.warmup_lr, decay_epochs=args.decay_epochs, decay_lr=args.decay_lr, decay_mode=args.decay_mode)
    
    if args.predict_score:
        L2_loss = torch.nn.MSELoss()
        for epi in range(args.epochs):
            md = utils.MeterDict()
            if not args.concise:
                print("Epochs[%03d/%03d] lr:%.7f"%(epi, args.epochs, optimizer.param_groups[0]['lr']))
            for mode, sel_loader in [("train", train_loader), ("val", val_loader)]:
                set_model(encoder, mode)
                all_logits=[]
                all_y_preds1=[]
                all_y_preds2=[]
                all_y_gt=[]
                stl_type_gt=[]
                all_acc=[]
                all_pos_acc=[]
                all_neg_acc=[]
                for bi, batch in enumerate(sel_loader):
                    eta.update()
                    index_tensors, stl_embeds, ego_states, us, sa_trajectories, stl_str, sa_par, stl_i_tensors, stl_type_i_tensors = parse_batch(batch)
                    
                    # normalization
                    trajs2d = sa_trajectories[:, :, :2].reshape(sa_trajectories.shape[0], -1)
                    traj_embed = encoder.predict_score_head(trajs2d) 
                    traj_embed = traj_embed / torch.clip(torch.norm(traj_embed, dim=-1, keepdim=True), min=1e-4)
                    
                    stl_embed = encoder(None, stl_embeds) 
                    stl_embed = stl_embed / torch.clip(torch.norm(stl_embed, dim=-1, keepdim=True), min=1e-4)
                    logits_pred = torch.mm(stl_embed, traj_embed.t())
                    batch_size = traj_embed.size(0)
                    BARRIER=False
                    GROUP=False
                    if BARRIER:
                        group_labels = stl_i_tensors.unsqueeze(1)  # Shape: (batch_size, 1)
                        pos_mask = group_labels == (group_labels.t())  # Positive mask: (batch_size, batch_size)
                        neg_mask = torch.logical_not(pos_mask)
                        pos_loss = torch.mean(torch.nn.ReLU()(0.5 - logits_pred[pos_mask]))
                        neg_loss = torch.mean(torch.nn.ReLU()(logits_pred[neg_mask] + 0.5))
                        
                        pos_acc = torch.mean((logits_pred[pos_mask]>0.5).float())
                        neg_acc = torch.mean((logits_pred[neg_mask]<-0.5).float())
                        all_acc.append((pos_acc+neg_acc).item()/2)
                        all_pos_acc.append(pos_acc.item())
                        all_neg_acc.append(neg_acc.item())
                        loss = (pos_loss+neg_loss)/2
                        
                    elif GROUP:                        
                        group_labels = stl_i_tensors.unsqueeze(1)  # Shape: (batch_size, 1)
                        pos_mask = group_labels == group_labels.t()  # Positive mask: (batch_size, batch_size)                        
                        log_probs_x_to_y = F.log_softmax(logits_pred, dim=1)
                        log_probs_y_to_x = F.log_softmax(logits_pred.t(), dim=1)

                        # Positive pairs log probabilities
                        positive_log_probs_x_to_y = log_probs_x_to_y[pos_mask]
                        positive_log_probs_y_to_x = log_probs_y_to_x[pos_mask]

                        # Compute loss: average over positive pairs
                        loss1 = -positive_log_probs_x_to_y.mean()
                        loss2 = -positive_log_probs_y_to_x.mean()
                        loss = (loss1 + loss2)/2
                    
                    else:
                        # (B, K)
                        # (B, K)
                        # sim_mat = traj_embed @ (stl_embed.T)
                        y_gt = labels = torch.arange(batch_size, device=traj_embed.device)
                        loss1 = F.cross_entropy(logits_pred, labels)
                        loss2 = F.cross_entropy(logits_pred.T, labels)
                        loss = (loss1 + loss2)/2
                    
                    if mode=="train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    if BARRIER:
                        md.update("%s_loss"%(mode), loss.item())
                        md.update("%s_loss_pos"%(mode), pos_loss.item())
                        md.update("%s_loss_neg"%(mode), neg_loss.item())
                        if bi % args.print_freq == 0 or bi == len(sel_loader)-1:
                            print("Pretrain-epoch:%04d/%04d %s [%04d/%04d] loss:%.3f(%.3f)  pos_loss:%.3f(%.3f)  neg_loss:%.3f(%.3f)"%(
                                epi, args.epochs, mode.upper(), bi, len(sel_loader), 
                                md["%s_loss"%(mode)], md("%s_loss"%(mode)),
                                md["%s_loss_pos"%(mode)], md("%s_loss_pos"%(mode)),
                                md["%s_loss_neg"%(mode)], md("%s_loss_neg"%(mode)),
                            ))
                        
                    elif GROUP:
                        if bi % args.print_freq == 0 or bi == len(sel_loader)-1:
                            print("Pretrain-epoch:%04d/%04d %s [%04d/%04d] loss:%.3f(%.3f)"%(
                                epi, args.epochs, mode.upper(), bi, len(sel_loader), 
                                md["%s_loss"%(mode)], md("%s_loss"%(mode))))
                        
                    else:
                        y_pred1 = torch.argmax(logits_pred, dim=1)  # for STL-encoder
                        y_pred2 = torch.argmax(logits_pred, dim=0)  # for traj-encoder
                        acc1 = torch.mean((y_pred1==labels).float())
                        acc2 = torch.mean((y_pred2==labels).float())
                        # all_logits.append(logits_pred)
                        all_y_preds1.append(y_pred1)
                        all_y_preds2.append(y_pred2)
                        all_y_gt.append(y_gt)
                        stl_type_gt.append(stl_type_i_tensors)
                        
                        md.update("%s_loss"%(mode), loss.item())
                        md.update("%s_loss1"%(mode), loss1.item())
                        md.update("%s_loss2"%(mode), loss2.item())
                        md.update("%s_acc"%(mode), (acc1.item()+acc2.item())/2)
                        md.update("%s_acc1"%(mode), acc1.item())
                        md.update("%s_acc2"%(mode), acc2.item())
                        if bi % args.print_freq == 0 or bi == len(sel_loader)-1:
                            print("Pretrain-epoch:%04d/%04d %s [%04d/%04d] loss:%.3f(%.3f) acc:%.3f(%.3f) acc1:%.3f(%.3f) acc1:%.3f(%.3f)"%(
                                epi, args.epochs, mode.upper(), bi, len(sel_loader), 
                                md["%s_loss"%(mode)], md("%s_loss"%(mode)), 
                                md["%s_acc"%(mode)], md("%s_acc"%(mode)),
                                md["%s_acc1"%(mode)], md("%s_acc1"%(mode)),
                                md["%s_acc2"%(mode)], md("%s_acc2"%(mode))
                            ))
                
                if BARRIER:
                    print("%s  epi-%d  ACC: %.4f %.4f %.4f"%(mode, epi, np.mean(all_acc), np.mean(all_pos_acc), np.mean(all_neg_acc)))
                elif GROUP:
                    do_nothing=1
                else:
                    # all_logits = torch.cat(all_logits).detach()
                    all_y_preds1 = torch.cat(all_y_preds1).detach()
                    all_y_preds2 = torch.cat(all_y_preds2).detach()
                    all_y_gt = torch.cat(all_y_gt).detach()
                    stl_type_gt = torch.cat(stl_type_gt).detach()
                    accuracy1 = torch.mean((all_y_preds1==all_y_gt).float())
                    accuracy2 = torch.mean((all_y_preds2==all_y_gt).float())
                    
                    acc_type0=torch.mean((all_y_preds1==all_y_gt).float() * ((stl_type_gt==0).float()))/torch.clip(torch.mean((stl_type_gt==0).float()),1e-4)
                    acc_type1=torch.mean((all_y_preds1==all_y_gt).float() * ((stl_type_gt==1).float()))/torch.clip(torch.mean((stl_type_gt==1).float()),1e-4)
                    acc_type2=torch.mean((all_y_preds1==all_y_gt).float() * ((stl_type_gt==2).float()))/torch.clip(torch.mean((stl_type_gt==2).float()),1e-4)
                    acc_type3=torch.mean((all_y_preds1==all_y_gt).float() * ((stl_type_gt==3).float()))/torch.clip(torch.mean((stl_type_gt==3).float()),1e-4)
                    
                    acc_type0_=torch.mean((all_y_preds2==all_y_gt).float() * ((stl_type_gt==0).float()))/torch.clip(torch.mean((stl_type_gt==0).float()),1e-4)
                    acc_type1_=torch.mean((all_y_preds2==all_y_gt).float() * ((stl_type_gt==1).float()))/torch.clip(torch.mean((stl_type_gt==1).float()),1e-4)
                    acc_type2_=torch.mean((all_y_preds2==all_y_gt).float() * ((stl_type_gt==2).float()))/torch.clip(torch.mean((stl_type_gt==2).float()),1e-4)
                    acc_type3_=torch.mean((all_y_preds2==all_y_gt).float() * ((stl_type_gt==3).float()))/torch.clip(torch.mean((stl_type_gt==3).float()),1e-4)
                    
                    print(mode, epi, "ACC: %.4f %.4f (%.4f %.4f %.4f %.4f) (%.4f %.4f %.4f %.4f)"%(
                        accuracy1.item(), accuracy2.item(),
                        acc_type0.item(), acc_type1.item(), acc_type2.item(), acc_type3.item(), 
                        acc_type0_.item(), acc_type1_.item(), acc_type2_.item(), acc_type3_.item(), 
                    ))
            scheduler.step()
            utils.save_model_freq_last(encoder.state_dict(), args.model_dir, epi, args.save_freq, args.epochs)
                    
    
    if args.with_predict_head:
        CE_loss = torch.nn.CrossEntropyLoss()
        for epi in range(args.epochs):
            md = utils.MeterDict()
            if not args.concise:
                print("Epochs[%03d/%03d] lr:%.7f"%(epi, args.epochs, optimizer.param_groups[0]['lr']))
            for mode, sel_loader in [("train", train_loader), ("val", val_loader)]:
                set_model(encoder, mode)
                
                all_logits=[]
                all_y_preds=[]
                all_y_gt=[]
                for bi, batch in enumerate(sel_loader):
                    eta.update()
                    index_tensors, stl_embeds, ego_states, us, sa_trajectories, stl_str, sa_par, stl_i_tensors, stl_type_i_tensors = parse_batch(batch)
                    # NCE loss?
                    # self-recon loss?
                    y_gt = stl_type_i_tensors
                    
                    embedding = encoder(None, stl_embeds)
                    logits_pred = encoder.predict(embedding)
                    y_pred = torch.argmax(logits_pred, dim=-1)
                    loss = CE_loss(logits_pred, y_gt)

                    if mode=="train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                    acc = torch.mean((y_pred==y_gt).float())
                    all_logits.append(logits_pred)
                    all_y_preds.append(y_pred)
                    all_y_gt.append(y_gt)
                    
                    md.update("%s_loss"%(mode), loss.item())
                    md.update("%s_acc"%(mode), acc.item())
                    if bi % args.print_freq == 0 or bi == len(sel_loader)-1:
                        print("Pretrain-epoch:%04d/%04d %s [%04d/%04d] loss:%.3f(%.3f)  acc:%.3f(%.3f)"%(
                            epi, args.epochs, mode.upper(), bi, len(sel_loader), 
                            md["%s_loss"%(mode)], md("%s_loss"%(mode)), 
                            md["%s_acc"%(mode)], md("%s_acc"%(mode))
                        ))
                    
                all_logits = torch.cat(all_logits).detach()
                all_y_preds = torch.cat(all_y_preds).detach()
                all_y_gt = torch.cat(all_y_gt).detach()
                accuracy = torch.mean((all_y_preds==all_y_gt).float())
                print(mode, epi, "ACC: %.4f | 0:%.4f 1:%.4f 2:%.4f 3:%.4f"%(
                    accuracy.item(), 
                    torch.mean((all_y_preds==all_y_gt).float() * ((all_y_gt==0).float()))/torch.clip(torch.mean((all_y_gt==0).float()),1e-4),
                    torch.mean((all_y_preds==all_y_gt).float() * ((all_y_gt==1).float()))/torch.clip(torch.mean((all_y_gt==1).float()),1e-4),
                    torch.mean((all_y_preds==all_y_gt).float() * ((all_y_gt==2).float()))/torch.clip(torch.mean((all_y_gt==2).float()),1e-4),
                    torch.mean((all_y_preds==all_y_gt).float() * ((all_y_gt==3).float()))/torch.clip(torch.mean((all_y_gt==3).float()),1e-4),
                ))
            
            scheduler.step()
            utils.save_model_freq_last(encoder.state_dict(), args.model_dir, epi, args.save_freq, args.epochs)
    return

def train_score_predictor(score_predictor, tuple_data, train_loader, val_loader, CA, stat_min, stat_max, stat_mean, stat_std, par_indices):
    demo_list, stl_data_list, simple_stl_list, real_stl_list, obj_list, file_list, stl_str_list, type_list, cache = tuple_data    
    score_predictor = score_predictor.to(device)
    optimizer = torch.optim.Adam(score_predictor.parameters(), lr=args.lr)

    eta = utils.EtaEstimator(start_iter=0, end_iter=args.epochs * len(train_loader), )
    
    scheduler = utils.create_custom_lr_scheduler(optimizer, 
                                                 warmup_epochs=args.warmup_epochs, 
                                                 warmup_lr=args.warmup_lr, 
                                                 decay_epochs=args.decay_epochs, 
                                                 decay_lr=args.decay_lr, decay_mode=args.decay_mode)
    
    stl_acc_log_list = []
    
    for epi in range(args.epochs):
        md = utils.MeterDict()
        
        MINI_BATCH_N = 8
        N_NOISE = 2
        if epi % 5 == 0 or epi == args.epochs-1:
            stl_acc_log_list.append({"epoch":epi, "train":0, "val":0})
        if not args.concise:
            print("Epochs[%03d/%03d] lr:%.7f"%(epi, args.epochs, optimizer.param_groups[0]['lr']))
        for mode, sel_loader in [("train", train_loader), ("val", val_loader)]:
            set_model(score_predictor, mode)
            all_logits=[]
            all_y_preds=[]
            all_y_gt=[]
            
            acc_d = {0:[], 1:[], 2:[], 3:[]}
            
            for bi, batch in enumerate(sel_loader):
                eta.update()
                index_tensors, stl_embeds, ego_states, us, sa_trajectories, stl_str, sa_par, stl_i_tensors, stl_type_i_tensors = parse_batch(batch)
                
                BS = ego_states.shape[0]
                
                # get trajs
                trajs_gt = sa_trajectories.reshape(sa_trajectories.shape[0], args.horizon, args.observation_dim+args.action_dim)[:, :, :2]

                scores_gt_list=[]

                noised_trajs = sa_trajectories + torch.randn_like(sa_trajectories)
                total_trajs = torch.stack([sa_trajectories, noised_trajs], dim=0) # (2, BS, NT, K)
                # get stls to get scores
                for iii in range(MINI_BATCH_N):
                    rec_i, stl_i, ego_i, sol_i, rel_stl_index, rel_ego_i, rel_sol_i = sel_loader.dataset.fs_list[index_tensors[iii].item()]
                    real_stl = real_stl_list[stl_i]
                    scores_gt = real_stl(total_trajs[:, iii], args.smoothing_factor)[:, 0]
                    scores_gt_list.append(scores_gt)
                scores_gt_list = torch.stack(scores_gt_list, dim=1)  # (2, MINIBATCH)
                y_gt = scores_gt_list

                # clip smaller one
                y_gt = torch.clip(y_gt, -1, 1)
                total_trajs = total_trajs.reshape(2*BS, total_trajs.shape[-2], total_trajs.shape[-1])
                normalized_sa_trajs = normalize_traj(total_trajs, stat_mean=stat_mean, stat_std=stat_std) 
                
                normalized_sa_trajs_flat = normalized_sa_trajs.reshape(2 * BS, -1)
                
                # need to have the same format as in real triannig trajs
                # (None), (MINIBATCH,) , (2, NT*K)
                stl_feat = get_encoding(score_predictor.encoder, None, stl_embeds, stl_str)
                y_pred = score_predictor.dual_forward(None, stl_embeds, normalized_sa_trajs_flat, MINI_BATCH_N, stl_feat=stl_feat)
                
                loss = torch.mean(torch.square(y_pred - y_gt))
                if mode=="train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                acc_all = (torch.sign(y_pred)==torch.sign(y_gt)).float()
                
                acc = torch.mean(acc_all)
                
                if epi % 5 == 0 or epi == args.epochs-1:
                    for iiii in range(acc_all.shape[0]):
                        stl_type_i = int(stl_type_i_tensors[iiii].item())
                        acc_d[stl_type_i].append(acc_all[iiii])
                
                md.update("%s_loss"%(mode), loss.item())
                md.update("%s_acc"%(mode), acc.item())
                if bi % args.print_freq == 0 or bi == len(sel_loader)-1:
                    print("Score Pred:%04d/%04d %s [%04d/%04d] loss:%.3f(%.3f)  acc:%.3f(%.3f)   dt:%s  elapsed:%s  ETA:%s"%(
                        epi, args.epochs, mode.upper(), bi, len(sel_loader), 
                        md["%s_loss"%(mode)], md("%s_loss"%(mode)), md["%s_acc"%(mode)], md("%s_acc"%(mode)),
                        eta.interval_str(), eta.elapsed_str(), eta.eta_str()
                    ))
            
            if epi % 5 == 0 or epi == args.epochs-1:
                stl_acc_log_list[-1][mode] = [md("%s_acc"%(mode)), mean_func(acc_d[0]), mean_func(acc_d[1]), mean_func(acc_d[2]), mean_func(acc_d[3]),]
        scheduler.step()
        utils.save_model_freq_last(score_predictor.state_dict(), args.model_dir, epi, args.save_freq, args.epochs)
        
    np.savez("%s/stl_pred_accs.npz"%(args.exp_dir_full), data=stl_acc_log_list)
    return


def mean_func(x):
    if len(x)==0:
        return 0
    else:
        return np.mean([utils.to_np(xx) for xx in x])

def eval_function_simple(us, ego_states_tj, gen_traj_func, state_dim, real_stl):
    us = torch.from_numpy(us).float().reshape(1, args.nt, 2)
    trajs = gen_traj_func(ego_states_tj, us, dt=args.dt).reshape(1, args.nt+1, state_dim)
    stl_score = real_stl(trajs, args.smoothing_factor)[:, :1]
    acc = (stl_score>0).float()
    acc_mean = torch.mean(acc)
    loss_stl = torch.mean(torch.nn.ReLU()(args.stl_thres - stl_score)) * args.stl_weight
    loss_reg = torch.mean(torch.nn.ReLU()(us**2 - args.u_max**2)) * args.reg_weight
    loss_cost = torch.mean(us**2) * args.cost_weight * 0.3 + torch.mean(torch.diff(us, dim=-2)**2) * args.cost_weight + torch.mean(torch.diff(torch.diff(us, dim=-2),dim=-2)**2) * args.cost_weight
    loss_bdry = torch.mean(torch.nn.ReLU()(trajs**2 - args.x_max**2)/(args.x_max**2)) * args.bdry_weight
    loss = loss_stl + loss_reg + loss_cost + loss_bdry
    return loss.item(), us, trajs


def main():
    # setup the exp folder
    utils.setup_exp_and_logger(args, test=args.test, dryrun=args.dryrun) 
    
    # load the data
    print("Env:%s  use data %s ..."%(args.env, args.data_path))
    tuple_data = load_dataset()
    demo_list, stl_data_list, simple_stl_list, real_stl_list, obj_list, file_list, stl_str_list, type_list, cache = tuple_data
    seq_max_len = np.max([len(xxx) for xxx in stl_str_list])
    input_dim = 8  # gnn encoder: 8 node features
    
    train_loader, val_loader = get_data_loader(tuple_data, seq_max_len, input_dim)
    print("seq_max_len", seq_max_len)
    
    # simple env needs no renderer
    
    # get the GNN-conditioned diffusion/flow model ["gnn", "gru", "trans", "rnn_lstm", "rnn_gru", "tree_lstm"]
    # GNN embedding dim
    # type_i, t1, t2, is_left, x, y, z, r
    # Goal embedding dim
    # x, y, z, r, is_bs
    ego_state_dim = args.observation_dim
    encoder = GCN(input_dim, ego_state_dim, args).to(device)
    if args.cls_guidance:
        encoder_extra = GCN(input_dim, ego_state_dim, args).to(device)

    if args.sst:
        sst_encoder(encoder, tuple_data, train_loader, val_loader)
        return

    if args.mock_model:
        model = MockNet(args.horizon, args.data_dim, 
            cond_dim=args.condition_dim + ego_state_dim, 
            dim=args.tconv_dim,
            dim_mults=args.dim_mults, 
            attention=args.attention)
    elif args.mlp:
        print("use this")
        model = MLPNet(args.horizon, args.data_dim, 
            cond_dim=args.condition_dim + ego_state_dim, 
            dim=args.tconv_dim,
            dim_mults=args.dim_mults, 
            attention=args.attention)
    else:
        model = TemporalUnet(args.horizon, args.data_dim, 
            cond_dim=args.condition_dim + ego_state_dim, 
            dim=args.tconv_dim,
            dim_mults=args.dim_mults, 
            attention=args.attention,
            dropout=args.unet_dropout)

    if args.grad_nn:
        gen_model_class = GradNN
    elif args.flow:
        gen_model_class = GaussianFlow
    elif args.vae:
        gen_model_class = GaussianVAE
    else:
        gen_model_class = GaussianDiffusion
    
    diffuser = gen_model_class(
        model=model,
        horizon=args.horizon,
        observation_dim=args.data_dim-args.action_dim,
        action_dim=args.action_dim,
        n_timesteps=args.n_timesteps,
        loss_type=args.loss_type, # TODO here
        clip_denoised=args.clip_denoised,
        clip_value_min=-args.clip_value,
        clip_value_max=args.clip_value,
        encoder=encoder,
        transition_dim=args.data_dim,
    )
    
    if args.net_pretrained_path is not None:
        checkpoint = torch.load(os.path.join(utils.get_exp_dir(), utils.smart_path(args.net_pretrained_path)), weights_only=True)
        if args.load_unet:
            unet_checkpoints = {k: v for k, v in checkpoint.items() if k.startswith('model.')}
            diffuser.load_state_dict(unet_checkpoints, strict=False)
        elif args.load_encoder:
            if args.abs_name:
                enc_checkpoints = {k.replace("encoder.", ""): v for k, v in checkpoint.items() if k.startswith("encoder.")}
            else:
                enc_checkpoints = {k: v for k, v in checkpoint.items()}
            res = encoder.load_state_dict(enc_checkpoints, strict=False, )
            print(res.missing_keys)
        else:
            diffuser.load_state_dict(checkpoint)
    diffuser = diffuser.to(device)
    
    if args.train_encoder:
        trainable_names = [name for name, param in diffuser.named_parameters() if not name.startswith('model.')]
        if args.train_unet_partial:
            trainable_names += [name for name, param in diffuser.named_parameters() \
                if name.startswith('model.cond_mlp') or (name.startswith("model.time_mlp")==False and "time_mlp" in name)]
        print("train with", trainable_names)
        trainable_params = [param for name, param in diffuser.named_parameters() if name in trainable_names]
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    elif args.finetune_backbone:
        trainable_names = [name for name, param in diffuser.named_parameters() if not name.startswith('model.')]
        finetune_names = [name for name, param in diffuser.named_parameters() if name.startswith('model.')]
        print("train with", trainable_names)
        print("finetune with", finetune_names)
        trainable_params = [param for name, param in diffuser.named_parameters() if name in trainable_names]
        finetune_params = [param for name, param in diffuser.named_parameters() if name in finetune_names]
        optimizer = torch.optim.Adam([
            {'params': trainable_params, 'lr': args.lr},                # Higher LR for encoder
            {'params': finetune_params, 'lr': args.lr*0.1}              # Lower LR for backbone
        ])
    else:
        optimizer = torch.optim.Adam(diffuser.parameters(), lr=args.lr)

    scheduler = utils.create_custom_lr_scheduler(optimizer, warmup_epochs=args.warmup_epochs, warmup_lr=args.warmup_lr, decay_epochs=args.decay_epochs, decay_lr=args.decay_lr, decay_mode=args.decay_mode)
    
    # data stats for simple env
    x_min, x_max, y_min, y_max = -5, 5, -5, 5
    stat_mean, stat_std = torch.Tensor([0, 0, 0, 0]).to(device), torch.Tensor([5, 5, 1, 1]).to(device)
    stat_mean_np = utils.to_np(stat_mean)
    stat_std_np = utils.to_np(stat_std)
    stat_min, stat_max = None, None
    mode = "meanstd"
    par_indices = None

    if args.set_detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    if args.train_classifier or args.cls_guidance:
        CA = None
        if args.cls_guidance:
            score_predictor = ScorePredictor(encoder_extra, args.condition_dim, args.horizon, args.data_dim, args).to(device)
        else:
            score_predictor = ScorePredictor(encoder, args.condition_dim, args.horizon, args.data_dim, args).to(device)
            train_score_predictor(score_predictor, tuple_data, train_loader, val_loader, CA, stat_min, stat_max, stat_mean, stat_std, par_indices)
            return
    
    if args.cls_path is not None and (args.train_classifier or args.cls_guidance):
        checkpoint_cls = torch.load(os.path.join(utils.get_exp_dir(), utils.smart_path(args.cls_path)))
        score_predictor.load_state_dict(checkpoint_cls)
    
    norm_func = norm_func_decorator(args, mode, stat_min, stat_max, stat_mean, stat_std)
    denorm_func = denorm_func_decorator(args, stat_min, stat_max, stat_mean, stat_std, par_indices)
    
    def loss_func(test_x, test_stl):
        dbg_infos = {}
        score = test_stl(test_x, args.smoothing_factor)[:, 0]
        loss = torch.mean(torch.nn.ReLU()(0.5-score))
        return loss, dbg_infos
    
    # test eval
    if args.test:
        md = utils.MeterDict()
        diffuser.eval()    
        stl_acc_log_list = [{"epoch":0, "train":0, "val":0}]
        eval_stat={"train":{"stl":{}}, "val":{"stl":{}}}
        
        # each batch is a dict, and contains all attributes
        res_d = {"train":[], "val":[]}
        res_d["meta"] = {"env":args.env, "encoder":args.encoder, "args":args, "exp_dir_full":args.exp_dir_full}
        viz_cnt = {"train":0, "val":0}
        eta = utils.EtaEstimator(0, args.num_evals*2)
        for mode, sel_loader in [("train", train_loader), ("val", val_loader)]:
            # test acc
            eval_scores = []
            t_avg_list=[]
            eval_scores_d = {}
            eval_split = mode
            if args.val_eval_only and mode=="train":
                continue
            if args.train_eval_only and mode=="val":
                continue
            for bi, batch in enumerate(sel_loader):
                if bi >= args.num_evals:
                    break
                eta.update()
                res_d[mode].append({})
                RES = res_d[mode][-1]
                
                index_tensors, stl_embeds, ego_states, us, sa_trajectories, stl_str, sa_par, stl_i_tensors, stl_type_i_tensors = parse_batch(batch)
                                
                compute_t1 = time.time()
                ego_states_normalized = None

                bs = sa_trajectories.shape[0]
                if False:  # placeholder - block below is the only path
                    pass
                    if args.env in ["dubins", "simple"]:
                        if args.env=="dubins":
                            state_dim = 4
                            action_dim = 2
                            gen_traj_func = generate_scene_v1.generate_trajectories_dubins
                        else:
                            state_dim = 2
                            action_dim = 2
                            gen_traj_func = generate_scene_v1.generate_trajectories
                        mdd = utils.MeterDict()
                        
                        diffused_trajs_list = []
                        for iii in range(bs):
                            rec_i, stl_i, ego_i, sol_i, rel_stl_index, rel_ego_i, rel_sol_i = sel_loader.dataset.fs_list[index_tensors[iii].item()]
                            real_stl = real_stl_list[stl_i]
                            stl_type = type_list[stl_i]
                            
                            args.stl_thres = 0.5
                            args.stl_weight = 1.0
                            args.reg_weight = 1.0
                            args.cost_weight = 20.0
                            args.bdry_weight = 5
                            args.x_max = 5
                            args.y_max = 5
                            args.omega_max = args.accel_max = 1
                            args.u_max = 1
                            ego_states_tj = ego_states[iii:iii+1].repeat(N_INITS, 1)
                            
                            if args.cma_es:
                                options = {
                                'seed': args.cma_es_seed if args.cma_es_seed is not None else args.seed, 
                                'bounds': [-1., 1.], 'maxiter': args.cma_es_niters, 'popsize': args.cma_es_popsize,
                                'verbose': 1,   # Suppress all output (-9 is the most quiet mode)
                                }
                                if args.env in ["dubins", "simple"]:
                                    initial_mean = np.random.rand(args.nt, 2).flatten() - 0.5
                                sigma = 0.5
                                es = cma.CMAEvolutionStrategy(initial_mean, sigma, options)
                                ego_states_tj_single = ego_states_tj[0:1]
                                while not es.stop():
                                    solutions = es.ask()
                                    fitness = [eval_function_simple(sol, ego_states_tj_single.cpu(), gen_traj_func, state_dim, real_stl)[0] for sol in solutions]
                                    es.tell(solutions, fitness)
                                    print(fitness)
                                best_solution = es.result.xbest
                                loss_item, us_optim, es_traj = eval_function_simple(best_solution, ego_states_tj_single.cpu(), gen_traj_func, state_dim, real_stl) # eval_function_simple(best_solution)
                                diffused_trajs_item = es_traj[0].to(device)
                            
                            elif args.cem:
                                n_iters = 100
                                n_elites = 10
                                n_popsize = 64
                                mean = 0
                                std = 1.0
                                
                                for iter_i in range(n_iters):
                                    us = mean + std * torch.randn(n_popsize, args.nt, 2).to(device)
                                    us = torch.clip(us, -1, 1)
                                    trajs = gen_traj_func(ego_states_tj, us, dt=args.dt, v_max=args.v_max).reshape(n_popsize, args.nt+1, state_dim)
                                    stl_score = real_stl(trajs, args.smoothing_factor)[:, :1]
                                    acc = (stl_score>0).float()
                                    acc_mean = torch.mean(acc)
                                    
                                    loss_stl_m = torch.nn.ReLU()(args.stl_thres - stl_score[:,0]) * args.stl_weight
                                    if args.env=="dubins":
                                        loss_reg_m = torch.mean(torch.nn.ReLU()(us[...,0]**2 - args.omega_max**2) * args.reg_weight + torch.nn.ReLU()(us[...,1]**2 - args.accel_max**2) * args.reg_weight, dim=-1)/2
                                        loss_cost_m = torch.mean(torch.mean(us**2, dim=-1),dim=-1) * args.cost_weight * 0.1
                                        loss_bdry_x_m = torch.mean(torch.nn.ReLU()(trajs[...,0]**2 - args.x_max**2)/(args.x_max**2), dim=-1) * args.bdry_weight
                                        loss_bdry_y_m = torch.mean(torch.nn.ReLU()(trajs[...,1]**2 - args.y_max**2)/(args.y_max**2), dim=-1) * args.bdry_weight
                                        loss_bdry_m = (loss_bdry_x_m + loss_bdry_y_m)/2
                                    else:
                                        loss_reg_m = torch.mean(torch.mean(torch.nn.ReLU()(us**2 - args.u_max**2),dim=-1), dim=-1)* args.reg_weight
                                        loss_cost_m = torch.mean(torch.mean(us**2, dim=-1),dim=-1) * args.cost_weight * 0.3 + \
                                            torch.mean(torch.mean(torch.diff(us, dim=-2)**2,dim=-1),dim=-1) * args.cost_weight + \
                                                torch.mean(torch.mean(torch.diff(torch.diff(us, dim=-2),dim=-2)**2,dim=-1),dim=-1) * args.cost_weight
                                        loss_bdry_m = torch.mean(torch.mean(torch.nn.ReLU()(trajs**2 - args.x_max**2)/(args.x_max**2),dim=-1),dim=-1) * args.bdry_weight
                                    loss_m = loss_stl_m + loss_reg_m + loss_cost_m + loss_bdry_m
                                    loss = torch.mean(loss_m)
                                    
                                    # Select elite samples with lowest loss
                                    elite_idxs = torch.argsort(loss_m)[:n_elites]
                                    elite_samples = us[elite_idxs]

                                    # Update mean and std based on elite samples
                                    mean = elite_samples.mean(dim=0)
                                    std = elite_samples.std(dim=0, unbiased=False) + 1e-6  # Small value to prevent collapse
                                
                                arg_stl_max_i = torch.argmax(stl_score, dim=0)[0]
                                diffused_trajs_item = trajs[arg_stl_max_i]
                                
                            else:
                                if args.env=="dubins":
                                    omegas = utils.uniform_tensor(-args.omega_max *.5, args.omega_max *.5, (N_INITS, args.nt, 1))
                                    accels = utils.uniform_tensor(-args.accel_max *.5, args.accel_max *.5, (N_INITS, args.nt, 1))
                                    us = torch.cat([omegas, accels], dim=-1).float().to(device).requires_grad_(True)
                                else:
                                    us = utils.uniform_tensor(-args.u_max *.5, args.u_max *.5, (N_INITS, args.nt, 2)).float().to(device).requires_grad_(True)

                                optimizer_tj = torch.optim.Adam([us], lr=args.trajopt_lr)
                                for iter_i in range(args.trajopt_niters):
                                    trajs = gen_traj_func(ego_states_tj, us, dt=args.dt, v_max=args.v_max).reshape(N_INITS, args.nt+1, state_dim)
                                    stl_score = real_stl(trajs, args.smoothing_factor)[:, :1]
                                    acc = (stl_score>0).float()
                                    acc_mean = torch.mean(acc)
                                    loss_stl = torch.mean(torch.nn.ReLU()(args.stl_thres - stl_score)) * args.stl_weight
                                    if args.env=="dubins":
                                        loss_reg = (torch.mean(torch.nn.ReLU()(us[...,0]**2 - args.omega_max**2)) * args.reg_weight + torch.mean(torch.nn.ReLU()(us[...,1]**2 - args.accel_max**2)) * args.reg_weight)/2
                                        loss_cost = torch.mean(us**2) * args.cost_weight * 0.1
                                        loss_bdry_x = torch.mean(torch.nn.ReLU()(trajs[...,0]**2 - args.x_max**2)/(args.x_max**2)) * args.bdry_weight
                                        loss_bdry_y = torch.mean(torch.nn.ReLU()(trajs[...,1]**2 - args.y_max**2)/(args.y_max**2)) * args.bdry_weight
                                        loss_bdry = (loss_bdry_x + loss_bdry_y)/2
                                    else:
                                        loss_reg = torch.mean(torch.nn.ReLU()(us**2 - args.u_max**2)) * args.reg_weight
                                        loss_cost = torch.mean(us**2) * args.cost_weight * 0.3 + torch.mean(torch.diff(us, dim=-2)**2) * args.cost_weight + torch.mean(torch.diff(torch.diff(us, dim=-2),dim=-2)**2) * args.cost_weight
                                        loss_bdry = torch.mean(torch.nn.ReLU()(trajs**2 - args.x_max**2)/(args.x_max**2)) * args.bdry_weight
                                    loss = loss_stl + loss_reg + loss_cost + loss_bdry
                                    
                                    optimizer_tj.zero_grad()
                                    loss.backward()
                                    optimizer_tj.step()
                                    
                                    mdd.update("loss", loss.item())
                                    mdd.update("loss_stl", loss_stl.item())
                                    mdd.update("loss_reg", loss_reg.item())
                                    mdd.update("loss_cost", loss_cost.item())
                                    mdd.update("acc", acc_mean.item())
                                    
                                    if (iter_i % 10 == 0 or iter_i ==args.trajopt_niters-1) and args.quiet==False:
                                        print("%s BS-%d iii-%04d || Trajopt-Iter:%04d/%04d loss:%.3f(%.3f) stl:%.3f(%.3f) reg:%.3f(%.3f) cost:%.3f(%.3f) acc:%.3f(%.3f)"%(
                                            mode.upper(), bi, iii, iter_i, args.trajopt_niters, mdd["loss"], mdd("loss"), mdd["loss_stl"], mdd("loss_stl"), 
                                            mdd["loss_reg"], mdd("loss_reg"), mdd["loss_cost"], mdd("loss_cost"), mdd["acc"], mdd("acc"), 
                                        ))
                                arg_stl_max_i = torch.argmax(stl_score, dim=0)[0]
                                diffused_trajs_item = trajs[arg_stl_max_i]
                            diffused_trajs_list.append(diffused_trajs_item)
                        diffused_trajs = torch.stack(diffused_trajs_list, dim=0)
                        
                    elif args.env == "panda":
                        a = 1
                        dof = 7
                        args.u_max = 1.0
                        mdd = utils.MeterDict()
                        print("Learning rate set to %.4f niters %d"%(args.trajopt_lr, args.trajopt_niters))
                        diffused_trajs_list = []
                        for iii in range(bs):
                            rec_i, stl_i, ego_i, sol_i, rel_stl_index, rel_ego_i, rel_sol_i = sel_loader.dataset.fs_list[index_tensors[iii].item()]
                            real_stl = real_stl_list[stl_i]
                            stl_type = type_list[stl_i]
                            
                            record = demo_list[rec_i]
                            obstacles_indices = record["obstacles_indices"]
                            curr_objects = record['objects']
                            avoid_loc_list = torch.tensor([curr_objects[coll_idx] for coll_idx in obstacles_indices]).to(device)
                            ego_states_tj = ego_states[iii:iii+1].repeat(N_INITS, 1)
                            us_raw = torch.randn(N_INITS, args.nt, dof).float().to(device).requires_grad_(True)
                            
                            if args.cem:
                                n_iters = 200
                                n_elites = 10
                                n_popsize = 64
                                SEP_CEM = False
                                mean = 0
                                std = 1.0
                                ego_states_tj = ego_states[iii:iii+1].repeat(n_popsize, 1)
                                
                                sigma = torch.ones(args.nt, dof).to(device)
                                mu = n_elites  # Number of elite samples
                                sep_cem_weights = torch.tensor([np.log(mu + 0.5) - np.log(iiiii + 1) for iiiii in range(mu)]).to(device)
                                sep_cem_weights /= sep_cem_weights.sum()  # Normalize weights
                                
                                for iter_i in range(n_iters):
                                    if SEP_CEM:
                                        us = mean + sigma * torch.randn(n_popsize, args.nt, dof).to(device)
                                    else:
                                        us = mean + std * torch.randn(n_popsize, args.nt, dof).to(device)
                                    us = torch.clip(us, -1, 1)
                                    trajs = get_trajectories(ego_states_tj, us, dof, args.dt)
                                    stl_x = panda_postproc(CA, trajs)
                                    seg_all_points = stl_x["points"]
                                    stl_scores = real_stl(stl_x, tau=500)[:, 0]
                                    acc_avg = torch.mean((stl_scores>0).float())
                                    stl_loss_m = torch.nn.ReLU()(0.5 - stl_scores)
                                    
                                    dense_point = interpolate_func(seg_all_points, N=4)
                                    if len(obstacles_indices)>0:
                                        avoid_obj_r = avoid_loc_list[:, 3]
                                        safe_r = 0.0
                                        avoid_loss_m = torch.sum(torch.sum(torch.sum(torch.nn.ReLU()(avoid_obj_r + safe_r - torch.norm(dense_point[..., None, :]-avoid_loc_list[:, 0:3],dim=-1)), dim=-1),dim=0),dim=-1)
                                    else:
                                        avoid_loss_m = 0 * stl_loss_m
                                    reg_loss = 0
                                    for i in range(dof):
                                        reg_loss_item1 = torch.nn.ReLU()(thmin[i] - trajs[..., i])
                                        reg_loss_item2 = torch.nn.ReLU()(trajs[..., i] - thmax[i])
                                        reg_loss_m = torch.mean(torch.mean(reg_loss + reg_loss_item1 + reg_loss_item2, dim=-1), dim=-1)

                                    above_mid_air_loss_m = torch.sum(torch.sum(torch.nn.ReLU()(0.8 - seg_all_points[[4,5,6], ..., 2]), dim=0), dim=-1) * 1
                                    above_table_loss_m = torch.sum(torch.sum(torch.nn.ReLU()(0.65 - seg_all_points[2:, ..., 2]), dim=0), dim=-1) * 10
                                    # (M, B, T+1, 3)
                                    smooth_loss_m = torch.mean(torch.mean(torch.square(us[:, 1:]-us[:, :-1]), dim=-1), dim=-1) * 0.2
                                    loss_m = stl_loss_m + reg_loss_m + above_mid_air_loss_m + above_table_loss_m + avoid_loss_m + smooth_loss_m
                                    loss = torch.mean(loss_m)

                                    # Select elite samples with lowest loss
                                    elite_idxs = torch.argsort(loss_m)[:n_elites]
                                    elite_samples = us[elite_idxs]

                                    # Update mean and std based on elite samples
                                    if SEP_CEM:
                                        new_mean = (sep_cem_weights.view(-1, 1, 1) * elite_samples).sum(dim=0)
                                        sigma = torch.sqrt(((sep_cem_weights.view(-1, 1, 1) * (elite_samples - new_mean) ** 2).sum(dim=0)) + 1e-6)
                                    else:
                                        mean = elite_samples.mean(dim=0)
                                        std = elite_samples.std(dim=0, unbiased=False) + 1e-6  # Small value to prevent collapse
                                    
                                print(iter_i, loss.item(), acc_avg.item())
                                arg_stl_max_i = torch.argmax(stl_scores, dim=0)
                                diffused_trajs_item = trajs[arg_stl_max_i]
                            else:
                                optimizer_tj = torch.optim.Adam([us_raw], lr=args.trajopt_lr)
                                for iter_i in range(args.trajopt_niters):
                                    factored_us = torch.tanh(us_raw * 1) * args.u_max
                                    trajs = get_trajectories(ego_states_tj, factored_us, dof, args.dt)
                                    stl_x = panda_postproc(CA, trajs)
                                    seg_all_points = stl_x["points"]
                                    stl_scores = real_stl(stl_x, tau=500)[:, 0]
                                    acc_avg = torch.mean((stl_scores>0).float())
                                    stl_loss = torch.mean(torch.nn.ReLU()(0.5 - stl_scores))
                                    dense_point = interpolate_func(seg_all_points, N=4)
                                    if len(obstacles_indices)>0:
                                        avoid_obj_r = avoid_loc_list[:, 3]
                                        safe_r = 0.0
                                        avoid_loss = torch.sum(torch.nn.ReLU()(avoid_obj_r + safe_r - torch.norm(dense_point[..., None, :]-avoid_loc_list[:, 0:3],dim=-1)))
                                    else:
                                        avoid_loss = 0 * stl_loss
                                    reg_loss = 0
                                    for i in range(dof):
                                        reg_loss_item1 = torch.nn.ReLU()(thmin[i] - trajs[..., i])
                                        reg_loss_item2 = torch.nn.ReLU()(trajs[..., i] - thmax[i])
                                        reg_loss = torch.mean(reg_loss + reg_loss_item1 + reg_loss_item2)

                                    above_mid_air_loss = torch.sum(torch.nn.ReLU()(0.8 - seg_all_points[[4,5,6], ..., 2])) * 1
                                    above_table_loss = torch.sum(torch.nn.ReLU()(0.65 - seg_all_points[2:, ..., 2])) * 10
                                    # (M, B, T+1, 3)
                                    smooth_loss = torch.mean(torch.square(factored_us[:, 1:]-factored_us[:, :-1])) * 0.2
                                    loss = stl_loss + reg_loss + above_mid_air_loss + above_table_loss + avoid_loss + smooth_loss
                                    
                                    optimizer_tj.zero_grad()
                                    loss.backward()
                                    optimizer_tj.step()

                                    if iter_i % 10 == 0 and args.quiet==False:
                                        print("%s BS-%05d iiii:%d Iter-%05d Loss:%.4f  stl:%.4f acc:%.3f reg:%.4f aboe:%.4f air:%.4f avoid:%.4f smooth:%.4f"%(
                                            mode.upper(), bi, iii, iter_i, loss.item(), stl_loss.item(), acc_avg.item(), reg_loss.item(), 
                                            above_table_loss.item(), above_mid_air_loss.item(), avoid_loss.item(),
                                            smooth_loss.item()
                                            ))
                                
                                arg_stl_max_i = torch.argmax(stl_scores, dim=0)
                                diffused_trajs_item = trajs[arg_stl_max_i]
                            diffused_trajs_list.append(diffused_trajs_item)
                # learning-based
                conditions = get_encoding(encoder, ego_states, stl_embeds, stl_str)
                if args.test_muls is not None:
                    conditions = conditions[:, None].repeat(1, args.test_muls, 1).reshape(bs * args.test_muls, conditions.shape[-1])

                if args.guidance or args.cls_guidance:
                    tmp_real_stl_list=[]
                    for iii in range(ego_states.shape[0]):
                        rec_i, stl_i, ego_i, sol_i, rel_stl_index, rel_ego_i, rel_sol_i = sel_loader.dataset.fs_list[index_tensors[iii].item()]
                        real_stl = real_stl_list[stl_i]
                        tmp_real_stl_list.append(real_stl)
                    guidance_data = {"args":args, "denorm": denorm_func, "norm_func":norm_func,
                                    "loss_func":loss_func, "real_stl_list":tmp_real_stl_list}
                    if args.cls_guidance:
                        guidance_data["score_predictor"] = score_predictor
                        guidance_data["stl_embeds"] = stl_embeds
                else:
                    guidance_data = None

                diffused_trajs = get_denoising_results(diffuser, conditions, stat_mean, stat_std, stat_min, stat_max, par_indices, in_painting=ego_states_normalized,
                                                    guidance_data=guidance_data)
                
                compute_t2 = time.time()
                RES["t"] = compute_t2 - compute_t1
                RES["scores"] = []
                RES["acc"] = []
                RES['index'] = []
                RES['rec_i'] = []
                RES['stl_i'] = []
                RES['stl_type_i'] = []
                RES['trajs'] = []

                t_avg_list.append(RES["t"])
                for iii in range(ego_states.shape[0]):
                    rec_i, stl_i, ego_i, sol_i, rel_stl_index, rel_ego_i, rel_sol_i = sel_loader.dataset.fs_list[index_tensors[iii].item()]
                    real_stl = real_stl_list[stl_i]
                    stl_type = type_list[stl_i]
                    RES['index'].append(index_tensors[iii].item())
                    RES['rec_i'].append(rec_i)
                    RES['stl_i'].append(stl_i)
                    RES['stl_type_i'].append(stl_type)

                    if stl_i not in eval_stat[eval_split]["stl"]:
                        eval_stat[eval_split]["stl"][stl_i] = 0
                    eval_stat[eval_split]["stl"][stl_i]+=1
                    max_stl_i = 0
                    if args.test_muls is not None:
                        score = real_stl(diffused_trajs[iii*args.test_muls:(iii+1)*args.test_muls], args.smoothing_factor)[:, 0]
                        score, max_stl_i = torch.max(score, dim=0)
                        score = score[None]
                    else:
                        score = real_stl(diffused_trajs[iii:iii+1], args.smoothing_factor)[:, 0]

                    eval_scores.append(int(score.item()>0))
                    if stl_type not in eval_scores_d:
                        eval_scores_d[stl_type] = []
                    eval_scores_d[stl_type].append(int(score.item()>0))

                    RES["scores"].append(score.item())
                    RES["acc"].append(int(score.item()>0))
                    if args.test_muls is not None:
                        best_traj = diffused_trajs[iii*args.test_muls+max_stl_i]
                    else:
                        best_traj = diffused_trajs[iii]

                    RES['trajs'].append(utils.to_np(best_traj))

                RES['trajs'] = np.stack(RES['trajs'], axis=0)
                    
                print("%-5s [%03d/%03d] acc:%d (%.3f) runtime:%.3f(%.3f)   dt:%s  Elapsed:%s  ETA:%s"%(
                    mode.upper(), bi, args.num_evals, RES["acc"][-1], np.mean(eval_scores), RES["t"], np.mean(t_avg_list),
                    eta.interval_str(), eta.elapsed_str(), eta.eta_str()
                    ))
            
                # visualization
                n_viz_trajs_max = args.n_viz_trajs_max
                
                diffused_trajs_np = utils.to_np(diffused_trajs)
                gt_trajs_np = utils.to_np(sa_trajectories.reshape(sa_trajectories.shape[0], args.horizon, args.observation_dim+args.action_dim)[:, :, :2])
                    
                for mini_i in range(index_tensors.shape[0]):
                    if viz_cnt[mode] >= args.max_viz:
                        continue
                    viz_cnt[mode]+=1
                    
                    rec_i, stl_i, ego_i, sol_i, rel_stl_index, rel_ego_i, rel_sol_i = sel_loader.dataset.fs_list[index_tensors[mini_i].item()]
                    the_record = demo_list[rec_i]
                    simp_stl = simple_stl_list[stl_i]
                    real_stl = real_stl_list[stl_i]
                    objects_d = obj_list[stl_i]
                    fig = plt.figure(figsize=(12, 4))
                    plt.subplot(1, 2, 1)
                    ax = plt.gca()
                    # scene
                    for obj_i, obj_keyid in enumerate(objects_d):
                        obj_d = objects_d[obj_keyid]
                        ax.add_patch(Circle([obj_d["x"], obj_d["y"]], radius=obj_d["r"], color="royalblue" if obj_d["r"]<0.75 else "gray", alpha=0.5))
                        plt.text(obj_d["x"], obj_d["y"], s="%d"%(obj_keyid))

                    # traj
                    plt.scatter(gt_trajs_np[mini_i, 0, 0], gt_trajs_np[mini_i, 0, 1], color="purple", s=64, zorder=1000)
                    plt.plot(gt_trajs_np[mini_i, :, 0], gt_trajs_np[mini_i, :, 1], color="green", alpha=0.3, linewidth=4, zorder=999)
                    if args.test_muls is not None:
                        for iii in range(min(n_viz_trajs_max, args.test_muls)):
                            # plot est trajs
                            m_idx = mini_i * args.test_muls + iii
                            plt.scatter(diffused_trajs_np[m_idx, 0, 0], diffused_trajs_np[m_idx, 0,1], color="orange", alpha=0.05, s=48, zorder=1000)
                            plt.plot(diffused_trajs_np[m_idx, :,0], diffused_trajs_np[m_idx, :,1], linewidth=2, color="brown", alpha=0.3)

                        plt.scatter(RES['trajs'][mini_i, 0, 0], RES['trajs'][mini_i, 0, 1], color="orange", alpha=0.5, s=48, zorder=20)
                        plt.plot(RES['trajs'][mini_i, :, 0], RES['trajs'][mini_i, :, 1], linewidth = 2, color="brown", alpha=0.8, zorder=1500)

                    else:
                        # plot est trajs
                        plt.scatter(diffused_trajs_np[mini_i, 0, 0], diffused_trajs_np[mini_i, 0,1], color="orange", alpha=0.05, s=48, zorder=1000)
                        plt.plot(diffused_trajs_np[mini_i, :,0], diffused_trajs_np[mini_i, :,1], linewidth=2, color="brown", alpha=0.3)

                    plt.axis("scaled")
                    plt.xlim(x_min, x_max)
                    plt.ylim(y_min, y_max)

                    # traj-tree
                    plt.subplot(1, 2, 2)
                    generate_scene_v1.plot_tree(simp_stl)
                    plt.xticks([])
                    plt.yticks([])
                    utils.plt_save_close("%s/viz_%s_b%04d_%d.png"%(args.viz_dir, mode, bi, mini_i))

            stl_acc_log_list[-1][eval_split]=np.mean(eval_scores)
            for stl_type in sorted(eval_scores_d.keys()):
                stl_acc_log_list[-1][eval_split+"_%d"%stl_type] = np.mean(eval_scores_d[stl_type])
            print("%s STL Acc:%.3f"%(eval_split, np.mean(eval_scores)))
            if bi % 20==0:
                np.savez("%s/stl_acc.npz"%(args.exp_dir_full), data=stl_acc_log_list)
                np.savez("%s/results.npz"%(args.exp_dir_full), data=res_d)
            
        train_acc_str = " ".join(["%s:%.4f"%(stl_type[5:], stl_acc_log_list[-1][stl_type]) for stl_type in stl_acc_log_list[-1] if "train" in stl_type])
        val_acc_str = " ".join(["%s:%.4f"%(stl_type[5:], stl_acc_log_list[-1][stl_type]) for stl_type in stl_acc_log_list[-1] if "val" in stl_type])
        print("STAT train:%d stls, %d trajs (%s) | val:%d stls, %d trajs (%s)"%(
            len(eval_stat["train"]["stl"]), sum(eval_stat["train"]["stl"].values()), train_acc_str,
            len(eval_stat["val"]["stl"]), sum(eval_stat["val"]["stl"].values()), val_acc_str,
        ))
        np.savez("%s/stl_acc.npz"%(args.exp_dir_full), data=stl_acc_log_list)
        
        # save the time profile dataz
        np.savez("%s/results.npz"%(args.exp_dir_full), data=res_d)
        return

    # training loop
    else:
        stl_acc_log_list=[]
        train_step = 0
        start_epi = 0
        CE_loss = torch.nn.CrossEntropyLoss()
        
        if args.rebase is not None:
            start_epi = args.rebase
            train_step = start_epi * len(train_loader)
        eta = utils.EtaEstimator(start_iter=start_epi, end_iter=args.epochs * len(train_loader), )
        for epi in range(start_epi, args.epochs):
            md = utils.MeterDict()
            all_logits=[]
            all_y_preds=[]
            all_y_gt=[]
            if not args.concise:
                print("Epochs[%03d/%03d] lr:%.7f"%(epi, args.epochs, optimizer.param_groups[0]['lr']))    
            diffuser.train()
            for bi, batch in enumerate(train_loader):
                eta.update()
                if args.debug:
                    continue
                
                index_tensors, stl_embeds, ego_states, us, sa_trajectories, stl_str, sa_par, stl_i_tensors, stl_type_i_tensors = parse_batch(batch)
                        
                normalized_sa_trajs = normalize_traj(sa_trajectories, stat_mean=stat_mean, stat_std=stat_std)
                ego_states_normalized = None
                
                # PREPROC
                batch_size = normalized_sa_trajs.shape[0]
                                
                # MODEL
                conditions = get_encoding(encoder, ego_states, stl_embeds, stl_str)
                
                # COMPUTE THE LOSS
                if args.grad_nn:
                    diffused_trajs = get_denoising_results(diffuser, conditions, stat_mean, stat_std, stat_min, stat_max, par_indices)
                    grad_score_list=[]
                    grad_trajs = generate_scene_v1.generate_trajectories(diffused_trajs[:,0,:2], diffused_trajs[:,:,2:],dt=0.5)[:, :-1]
                    for iii in range(grad_trajs.shape[0]):
                        rec_i, stl_i, ego_i, sol_i, rel_stl_index, rel_ego_i, rel_sol_i = train_loader.dataset.fs_list[index_tensors[iii].item()]
                        real_stl = real_stl_list[stl_i]
                        stl_type = type_list[stl_i]
                        score = real_stl(grad_trajs[iii:iii+1], args.smoothing_factor)[:, 0]
                        grad_score_list.append(score)
                    grad_score_items = torch.stack(grad_score_list, dim=0)
                    loss = torch.mean(torch.nn.ReLU()(0.5-grad_score_items))
                    grad_acc = torch.mean((grad_score_items>0).float())
                else:
                    t = torch.randint(0, args.n_timesteps, (batch_size,), device=device).long()
                    noise = torch.randn_like(normalized_sa_trajs)
                    x_noisy = diffuser.q_sample(x_start=normalized_sa_trajs, t=t, noise=noise)
                    x_noisy = x_noisy.reshape(batch_size, args.horizon, args.data_dim)
                    noise = noise.reshape(batch_size, args.horizon, args.data_dim)
                
                    x_recon = diffuser.model(x_noisy, conditions, t)
                    if args.flow:
                        x_target = normalized_sa_trajs - noise
                    else:
                        x_target = noise
                    if args.xy_mask:
                        loss = torch.mean(torch.square(x_recon[..., :2] - x_target[..., :2]))
                    elif args.pure_l2_index is not None:
                        loss = torch.mean(torch.square(x_recon[..., :args.pure_l2_index] - x_target[..., :args.pure_l2_index]))
                    else:
                        loss = F.mse_loss(x_recon, x_target)
                
                loss = loss * args.loss_weight
                if args.add_gnn_loss:
                    group_labels = stl_i_tensors.unsqueeze(1)  # Shape: (batch_size, 1)
                    pos_mask = group_labels == group_labels.t()  # Positive mask: (batch_size, batch_size)
                    embedding = encoder(None, stl_embeds)
                    embedding = embedding / torch.clip(torch.norm(embedding, dim=-1, keepdim=True), min=1e-4)
                    logits_pred = torch.mm(embedding, embedding.t())
                    log_probs_x_to_y = F.log_softmax(logits_pred, dim=1)
                    log_probs_y_to_x = F.log_softmax(logits_pred.t(), dim=1)

                    # Positive pairs log probabilities
                    positive_log_probs_x_to_y = log_probs_x_to_y[pos_mask]
                    positive_log_probs_y_to_x = log_probs_y_to_x[pos_mask]

                    # Compute loss: average over positive pairs
                    loss1 = -positive_log_probs_x_to_y.mean()
                    loss2 = -positive_log_probs_y_to_x.mean()
                    loss = loss + (loss1 + loss2)/2
                
                add_str=""
                if args.with_predict_head:
                    embedding = encoder(None, stl_embeds)
                    y_gt = stl_type_i_tensors
                    logits_pred = encoder.predict(embedding)
                    y_pred = torch.argmax(logits_pred, dim=-1)
                    ce_loss = CE_loss(logits_pred, y_gt)
                    old_loss = loss
                    loss = old_loss + ce_loss
                
                    sst_acc = torch.mean((y_pred==y_gt).float())
                    all_logits.append(logits_pred)
                    all_y_preds.append(y_pred)
                    all_y_gt.append(y_gt)
                    md.update("acc", sst_acc.detach().item())
                    md.update("old_loss", old_loss.detach().item())
                    md.update("ce_loss", ce_loss.detach().item())
                    add_str=" | old_loss:%.3f(%.3f) ce_loss:%.3f(%.3f) acc:%.3f(%.3f) |"%(
                        md["old_loss"], md("old_loss"), md["ce_loss"], md("ce_loss"),
                        md["acc"], md("acc")
                    )

                if args.grad_nn:
                    md.update("acc", grad_acc.detach().item())
                    add_str=" | acc:%.3f(%.3f) |"%(md["acc"], md("acc"))
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                                
                md.update("loss", loss.detach().item())
                
                if bi % args.print_freq == 0:
                    print("Epochs[%03d/%03d] (%04d/%04d) loss:%.3f(%.3f) %s dt:%s Elapsed:%s ETA:%s"%(
                        epi, args.epochs, bi, len(train_loader), 
                        md["loss"], md("loss"), add_str, eta.interval_str(), eta.elapsed_str(), eta.eta_str()))
            
            scheduler.step()  
            
            utils.save_model_freq_last(diffuser.state_dict(), args.model_dir, epi, args.save_freq, args.epochs)
            
            # evaluate diffusion rollout performance (on both training and valid split)
            if (epi % args.eval_freq == 0 or epi == args.epochs-1) and args.debug==False and (args.skip_first_eval==False or epi!=0):
                diffuser.eval()
                stl_acc_log_list.append({"epoch":epi, "train":0, "val":0})
                eval_stat={"train":{"stl":{}}, "val":{"stl":{}}}
                for eval_split, sel_loader in [("train", train_loader), ("val",val_loader)]:
                    eval_scores = []
                    eval_scores_d = {}
                    for bi, batch in enumerate(sel_loader):
                        if bi >= args.num_evals:
                            continue
                        
                        index_tensors, stl_embeds, ego_states, us, sa_trajectories, stl_str, sa_par, stl_i_tensors, stl_type_i_tensors = parse_batch(batch)
                        
                        ego_states_normalized = None
                        conditions = get_encoding(encoder, ego_states, stl_embeds, stl_str)
                        
                        diffused_trajs = get_denoising_results(diffuser, conditions, stat_mean, stat_std, stat_min, stat_max, par_indices, in_painting=ego_states_normalized)
                        
                        if args.grad_nn:
                            grad_score_list=[]
                            grad_trajs = generate_scene_v1.generate_trajectories(diffused_trajs[:,0,:2], diffused_trajs[:,:,2:],dt=0.5)[:, :-1]
                            diffused_trajs = torch.cat([grad_trajs[:,:,:2], diffused_trajs[:,:,2:]], dim=-1)

                        for iii in range(diffused_trajs.shape[0]):
                            rec_i, stl_i, ego_i, sol_i, rel_stl_index, rel_ego_i, rel_sol_i = sel_loader.dataset.fs_list[index_tensors[iii].item()]
                            real_stl = real_stl_list[stl_i]
                            stl_type = type_list[stl_i]

                            if stl_i not in eval_stat[eval_split]["stl"]:
                                eval_stat[eval_split]["stl"][stl_i] = 0
                            eval_stat[eval_split]["stl"][stl_i]+=1

                            score = real_stl(diffused_trajs[iii:iii+1], args.smoothing_factor)[:, 0]
                            eval_scores.append(int(score.item()>0))
                            
                            if stl_type not in eval_scores_d:
                                eval_scores_d[stl_type] = []
                            eval_scores_d[stl_type].append(int(score.item()>0))
                    stl_acc_log_list[-1][eval_split]=np.mean(eval_scores)
                    for stl_type in sorted(eval_scores_d.keys()):
                        stl_acc_log_list[-1][eval_split+"_%d"%stl_type] = np.mean(eval_scores_d[stl_type])
                    print("%s STL Acc:%.3f"%(eval_split, np.mean(eval_scores)))
                train_acc_str = " ".join(["%s:%.4f"%(stl_type[5:], stl_acc_log_list[-1][stl_type]) for stl_type in stl_acc_log_list[-1] if "train" in stl_type])
                val_acc_str = " ".join(["%s:%.4f"%(stl_type[5:], stl_acc_log_list[-1][stl_type]) for stl_type in stl_acc_log_list[-1] if "val" in stl_type])
                print("STAT train:%d stls, %d trajs (%s) | val:%d stls, %d trajs (%s)"%(
                    len(eval_stat["train"]["stl"]), sum(eval_stat["train"]["stl"].values()),
                    train_acc_str,
                    len(eval_stat["val"]["stl"]), sum(eval_stat["val"]["stl"].values()),
                    val_acc_str,
                ))
                np.savez("%s/stl_acc.npz"%(args.exp_dir_full), data=stl_acc_log_list)
                
            if (epi % args.viz_freq == 0 or epi == args.epochs-1) and (args.skip_first_eval==False or epi!=0):
                if args.debug:
                    sel_loader = train_loader
                    max_b = 100
                else:
                    sel_loader = val_loader
                    max_b = 1
                for bi, batch in enumerate(sel_loader):
                    if args.debug:
                        print(bi, len(sel_loader))
                    if bi>=max_b:
                        continue
                    index_tensors, stl_embeds, ego_states, us, sa_trajectories, stl_str, sa_par, stl_i_tensors, stl_type_i_tensors = parse_batch(batch)
  
                    ego_states_normalized = None
                    conditions = get_encoding(encoder, ego_states, stl_embeds, stl_str)

                    n_viz = 100

                    diffused_trajs = get_denoising_results(diffuser, conditions[0:1].repeat(n_viz, 1), stat_mean, stat_std, stat_min, stat_max, par_indices, in_painting=ego_states_normalized)
                    diffused_trajs_np = utils.to_np(diffused_trajs)
                    gt_trajs_np = utils.to_np(sa_trajectories.reshape(sa_trajectories.shape[0], args.horizon, args.observation_dim+args.action_dim)[0, :, :2])
                    
                    # visualization
                    rec_i, stl_i, ego_i, sol_i, rel_stl_index, rel_ego_i, rel_sol_i = sel_loader.dataset.fs_list[index_tensors[0].item()]
                    the_record = demo_list[rec_i]
                    simp_stl = simple_stl_list[stl_i]
                    objects_d = obj_list[stl_i]
                    
                    plt.figure(figsize=(8, 8))
                    ax = plt.gca()
                    for obj_i, obj_keyid in enumerate(objects_d):
                        obj_d = objects_d[obj_keyid]
                        circ = Circle([obj_d["x"], obj_d["y"]], radius=obj_d["r"], color="royalblue" if obj_d["is_obstacle"]==False else "gray", alpha=0.5)
                        ax.add_patch(circ)
                        plt.text(obj_d["x"], obj_d["y"], s="%d"%(obj_keyid))

                    # plot gt trajs
                    plt.scatter(gt_trajs_np[0, 0], gt_trajs_np[0, 1], color="purple", s=64, zorder=1000)
                    plt.plot(gt_trajs_np[:, 0], gt_trajs_np[:, 1], color="green", alpha=0.3, linewidth=4, zorder=999)

                    # plot est trajs
                    plt.scatter(diffused_trajs_np[:, 0, 0], diffused_trajs_np[:, 0,1], color="orange", alpha=0.05, s=48, zorder=1000)
                    for viz_traj_i in range(n_viz):
                        plt.plot(diffused_trajs_np[viz_traj_i, :,0], diffused_trajs_np[viz_traj_i, :,1], linewidth=2, color="brown", alpha=0.3)
                    plt.axis("scaled")
                    plt.xticks([])
                    plt.yticks([])
                    plt.xlim(x_min, x_max)
                    plt.ylim(y_min, y_max)
                    utils.plt_save_close("%s/viz_epi%04d_b%d.png"%(args.viz_dir, epi, bi))
    
    return
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    add = parser.add_argument
    add("--seed", type=int, default=1007)
    add("--exp_name", '-e', type=str, default=None)
    add("--gpus", type=str, default="0")
    add("--epochs", type=int, default=1000)       # 10000
    add("--niters", type=int, default=10000)      # 10000
    add("--batch_size", '-b', type=int, default=256)
    add("--num_workers", type=int, default=8)     # 100000
    add("--print_freq", type=int, default=50)     # 50
    add("--viz_freq", type=int, default=200)     # 5000
    add("--eval_freq", type=int, default=200)    # 2000
    add("--save_freq", type=int, default=1000)    # 2000
    add("--lr", type=float, default=5e-4)         # 3e-4
    add("--cpu", action='store_true', default=False)
    add("--test", action='store_true', default=False)
    add("--dryrun", action='store_true', default=False)
    add("--net_pretrained_path", '-P', type=str, default=None, help="path for loading the neural net pretrained model")
    add("--load_unet", action='store_true', default=False)
    
    # diffuser configs
    add("--horizon", type=int, default=64)              # 10000
    add("--override_obs_act_dim", action='store_true', default=False)
    add("--observation_dim", type=int, default=2)      # 10000
    add("--action_dim", type=int, default=2)            # 10000
    add("--condition_dim", type=int, default=256)      # 10000
    add("--n_timesteps", type=int, default=100)        # 10000
    add("--loss_type", type=str, choices=["l1", "l2"], default='l2')    
    add("--mlp", action='store_true', default=False)
    add("--mlp_diffusion", action='store_true', default=False)
    add("--clip_denoised", action='store_true', default=False)
    add("--dim_mults", type=int, nargs="+", default=[1,2,4,8])
    add("--clip_value", type=float, default=3.0)
    
    # dataset configs
    add("--total_nums", type=int, default=None)
    add("--num_trials", type=int, default=100)
    add("--num_types", type=int, default=3)
    add("--data_path", type=str, default=None)
    add("--dt", type=float, default=0.5)
    
    # arch selection
    add("--encoder", type=str, choices=["gnn"], default='gnn')
    
    # GNN config
    add("--hiddens", type=int, nargs="+", default=[256, 256, 256, 256])
    add("--mlp_hiddens", type=int, nargs="+", default=[64, 64])
    add("--debug", action='store_true', default=False)
    add("--normalize", action='store_true', default=False)
    add("--smoothing_factor", type=float, default=None)
    add("--loss_weight", type=float, default=1.0)
    add("--rebase", type=int, default=None) 
    add("--no_shuffle", action='store_true', default=False)
    
    add('--unet_dropout', type=float, default=None, help='dropout applied to layers (0 = no dropout)')  # 0.2
    add("--set_detect_anomaly", action='store_true', default=False)
    add("--first_sat_init", action='store_true', default=False)
    add("--num_evals", type=int, default=10)
    add("--clip_max", type=int, default=None)
    add("--rand_aug_graph", action='store_true', default=False)
    add("--aug_graph", action='store_true', default=False)
    add("--rand_aug_eval", action='store_true', default=False)
    
    # lr schedule
    add("--scheduler", action='store_true', default=False)
    add("--warmup_epochs", type=int, default=0)
    add("--warmup_lr", type=float, default=None)
    add("--decay_epochs", type=int, default=900)
    add("--decay_lr", type=float, default=5e-5)
    add("--decay_mode", type=str, default="cosine", choices=["linear", "cosine"])
    
    # dataset modality
    add("--env", type=str, choices=['simple'], default='simple')
    
    # for debug
    add("--select_indices", type=int, nargs="+", default=None)
    add("--same_train_val", action='store_true', default=False)
    add("--xy_mask", action='store_true', default=False)
    add("--pure_l2", action='store_true', default=False)
    add("--pure_l2_index", type=int, default=None)
    add("--stat_decay", type=float, default=0.9)
    add("--attention", action='store_true', default=False)
    
    add("--tconv_dim", type=int, default=32)
    add("--flow", action='store_true', default=False)
    add("--arch", type=str, default="unet", choices=["unet", "trans"])
    add("--skip_first_eval", action='store_true', default=False)
    add("--ema", action='store_true', default=False)
    
    # GNN
    add("--aggr_type", type=int, default=0)
    add("--bidir", action='store_true', default=False)
    add("--gat", action='store_true', default=False)
    add("--residual", action='store_true', default=False)
    add("--post_residual", action='store_true', default=False)
    # for clipping
    add("--max_sol_clip", type=int, default=None) # how many sols per init we select
    
    add("--pretraining", action='store_true', default=False)
    
    add("--mock", action='store_true', default=False)
    add("--mock_model", action='store_true', default=False)
    add("--train_encoder", action='store_true', default=False)
    add("--finetune_backbone", action='store_true', default=False)
    add("--train_unet_partial", action='store_true', default=False)
    
    add("--sst", action='store_true', default=False)
    add("--filtered_types", type=int, nargs="+", default=None)
    add("--data_aug", action='store_true', default=False)
    add("--cross_child", action='store_true', default=False)
    add("--load_encoder", action='store_true', default=False)
    add("--with_predict_head", action='store_true', default=False)
    
    add("--predict_score", action='store_true', default=False)
    add("--abs_name", action='store_true', default=False)
    add("--type_ratios", type=float, nargs="+", default=None)
    
    add("--val_eval_only", '-E', action='store_true', default=False)
    add("--add_self_loops", action='store_true', default=False)
    add("--two_hop", action='store_true', default=False)
    add("--add_gnn_loss", action='store_true', default=False)
    add("--add_depth", action='store_true', default=False)
    add("--gcn_no_self_loops", action='store_true', default=False)
    add("--gin_conv", action='store_true', default=False)
    add("--dense_edges", action='store_true', default=False)
    add("--std_scale", type=float, default=None)
    
    add("--mock_indices", type=int, nargs="+", default=None)
    add("--mock_dup_times", type=int, default=None)
    add("--loc_perturb", action='store_true', default=False)
    add("--test_muls", type=int, default=None)
    
    # guidance configs
    add("--guidance", action='store_true', default=False)
    add("--guidance_lr", type=float, default=0.1)
    add("--guidance_scale", type=float, default=0.5)
    add("--guidance_steps", type=int, default=1)
    add("--guidance_before", type=int, default=None)
    
    add("--train_classifier", action='store_true', default=False)
    add("--traj_hiddens", type=int, nargs="+", default=[256, 256, 256, 256])
    add("--score_hiddens", type=int, nargs="+", default=[64, 64])
    
    add("--cls_guidance", action='store_true', default=False)
    add("--cls_path", '-C', type=str, default=None, help="path for loading the classifier")
    # add("--mlp", action='store_true', default=False)
    
    add("--quiet", action='store_true', default=False)
    
    add("--vae", action='store_true', default=False)
    
    add("--test_path", '-T', type=str, default=None, help="path for loading the neural net pretrained model")
    add("--fix", action='store_true', default=False)
    add("--max_viz", type=int, default=4)
    add("--flow_pattern", '-V', type=int, default=None)
    
    add("--load_cache", action='store_true', default=False)
    add("--train_eval_only", '-TE', action='store_true', default=False)
    add("--suffix", type=str, default=None)
    
    add("--grad_nn", action='store_true', default=False)
    add("--max_aug", type=int, default=4)
    
    add("--plot_data", action='store_true', default=False)
    add("--n_viz_trajs_max", type=int, default=64)
    
    add("--timeout_thres", type=float, default=60.0)
    add("--solver_timeout_thres", type=float, default=60.0)
    add("--concise", action='store_true', default=False)
    args = parser.parse_args()
    if args.cpu:
        device = "cpu"
    else:
        device = "cuda:0"
    
    if args.exp_name is None:
        args.exp_name = "%s_%s_F"%(args.env, args.encoder.split("_")[0])
        if args.seed != 1007:
            args.exp_name += "_%d"%(args.seed)
        print("setup exp name", args.exp_name)
    
    if "QTEST" in args.exp_name:
        args.concise=True    
    
    if args.test_path is not None:
        args.test = True
        args.net_pretrained_path = args.test_path
    
    
    if args.override_obs_act_dim==False:
        args.observation_dim = 2
        args.action_dim = 2

    if args.data_path is None:
        args.data_path = "data_0_simple"
    args.data_dim = args.observation_dim + args.action_dim
    
    if args.post_residual:
        args.residual=True
    
    if args.smoothing_factor is None:
        args.smoothing_factor = 10
    args.clip_denoised = True
    args.attention = True
    args.cached = True
    args.rand_aug_eval = True
    t1 = time.time()
    main()
    t2 = time.time()
    print("Finished in %.3f seconds"%(t2-t1))