import os
import numpy as np
import torch
import tqdm

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from os.path import join as ospj

import utils
import generate_scene_v1
from stl_to_seq_utils import rand_aug, hard_rand_aug, aug_graph, compute_tree_size


HASHI_D = {
    2:  tuple([1, 0, 0, 0, 0, 0, 0,]),  # negation
    0:  tuple([0, 1, 0, 0, 0, 0, 0,]),  # conjunction
    1:  tuple([0, 0, 1, 0, 0, 0, 0,]),  # disjunction
    5:  tuple([0, 0, 0, 1, 0, 0, 0,]),  # eventually
    6:  tuple([0, 0, 0, 0, 1, 0, 0,]),  # always
    7:  tuple([0, 0, 0, 0, 0, 1, 0,]),  # until
    8:  tuple([0, 0, 0, 0, 0, 0, 0,]),  # reach
}


def load_dataset(args):
    data_name = "data.npz"
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
            print("Clip the dataset from %d to %d => %d" % (original_len, args.clip_max, len(stl_data_all)))

    if args.mock_dup_times is not None:
        mock_indices = np.random.choice(args.mock_indices, args.mock_dup_times)
        stl_data_all = [stl_data_all[idx] for idx in mock_indices]

    type_stat = {}
    type_init_stat = {}
    type_sol_stat = {}
    demo_list = stl_data_all
    total_nums = len(demo_list)

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
    origin_tree_size = []
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
                cnt_d = {"n": 1}
                compute_tree_size(simple_stl_list[-1], cnt_d=cnt_d)
                origin_tree_size.append(cnt_d["n"])
                curr_cnt_stat = {'add': 0}
                simple_stl_list[-1] = aug_graph(simple_stl_list[-1], cfg={"tmax": args.horizon}, inplace=False, max_aug=args.max_aug, curr_cnt_stat=curr_cnt_stat)
                curr_add.append(curr_cnt_stat["add"])
                cnt_d = {"n": 1}
                compute_tree_size(simple_stl_list[-1], cnt_d=cnt_d)
                curr_tree_size.append(cnt_d["n"])
            else:
                cnt_d = {"n": 1}
                compute_tree_size(simple_stl_list[-1], cnt_d=cnt_d)
                curr_add.append(0)
                curr_tree_size.append(cnt_d["n"])
                origin_tree_size.append(cnt_d["n"])

            real_stl_list.append(
                generate_scene_v1.find_ap_in_lines(0, stl_dict={}, objects_d={}, lines=record["stl"], numpy=True, real_stl=True,
                                                   ap_mode="l2", until1=False))
            if args.word_format:
                real_stl_list[-1].update_format("word")
            type_list.append(record['stl_type_i'])
            stl_data_list.append(record["stl"])
            stl_str_list.append([0])  # gnn encoder uses dummy placeholder

        rec_i_2_stl_i[rec_i] = len(visited_stl_ids) - 1
        stl_i_2_rec_i[len(visited_stl_ids) - 1] = rec_i

    if not args.concise:
        print("TREESIZE original:%.3f add:%.3f new:%.3f" % (np.mean(origin_tree_size), np.mean(curr_add), np.mean(curr_tree_size)))
        print("Filter data...")

    file_list = []
    rel_stl_index = 0

    num_egos = 8
    num_inits = 8

    if args.type_ratios is not None:
        assert len(args.type_ratios) == 4
        ratios = np.array(args.type_ratios)
        ratios = ratios / np.sum(ratios)
        current_stl_cnt = {0: 0, 1: 0, 2: 0, 3: 0}
        expect_stl_quota = {0: 0, 1: 0, 2: 0, 3: 0}

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

        rel_ego_i = 0
        for ego_i in range(num_egos):
            rel_sol_i = 0
            for sol_i in range(num_inits):
                if score[ego_i, sol_i] > 0:
                    if args.type_ratios is not None:
                        if current_stl_cnt[stl_type_i] == expect_stl_quota[stl_type_i]:
                            continue
                        current_stl_cnt[stl_type_i] += 1

                    file_list.append([rec_i, rec_i, ego_i, sol_i, rel_stl_index, rel_ego_i, rel_sol_i])
                    rel_sol_i += 1

                if args.max_sol_clip is not None and rel_sol_i == args.max_sol_clip:
                    break

            if rel_sol_i > 0:
                rel_ego_i += 1

            type_sol_stat[stl_type_i] += rel_sol_i
            if args.first_sat_init and rel_ego_i == 1:
                break
            elif rel_ego_i == args.max_ego_clip:
                break
        type_init_stat[stl_type_i] += rel_ego_i
        type_stat[stl_type_i] += (rel_ego_i > 0)
        if rel_ego_i > 0:
            rel_stl_index += 1

    if not args.concise:
        print("*" * 10, "Stat", "*" * 10)
        if args.type_ratios is not None:
            print("Expect ratios: %.3f %.3f %.3f %.3f | Actual: %.3f %.3f %.3f %.3f" % (
                ratios[0], ratios[1], ratios[2], ratios[3],
                current_stl_cnt[0] / len(file_list), current_stl_cnt[1] / len(file_list),
                current_stl_cnt[2] / len(file_list), current_stl_cnt[3] / len(file_list),
            ))
        print("Sat cases:", type_stat)
        print("Sat inits:", type_init_stat)
        print("Sat sols:", type_sol_stat)

    if args.first_sat_init:
        print("LEN", total_nums, "SAT ratio", len(file_list) / (total_nums * num_inits))
    else:
        print("LEN", total_nums, "SAT ratio", len(file_list) / (total_nums * num_egos * num_inits))

    return demo_list, stl_data_list, simple_stl_list, real_stl_list, obj_list, file_list, stl_str_list, type_list, cache


def get_graph_stl_embed_from_tree(simple_stl, args, is_train=False):
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
                if node_type == 7:
                    queue.append([depth + 1, total_i, node.children[_i], 1 if _i == 0 else -1, 1 if _i == 1 else -1])
                else:
                    queue.append([depth + 1, total_i, node.children[_i], -1, -1])
                total_i += 1
        else:
            obj_x = node.obj_x
            obj_y = node.obj_y
            obj_z = node.obj_z
            obj_r = node.obj_r

        if args.hashi_gnn:
            node_feature = list(HASHI_D[node_type]) + [ta, tb, obj_x, obj_y, obj_z, obj_r, left_child]
        else:
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


class GSTLDataset(Dataset):
    def __init__(self, dataset, split, seq_max_len, embed_dim, args, shuffle=True):
        super().__init__(None, None, None, None)
        self.dataset = dataset
        self.split = split
        self.args = args
        _, _, _, _, _, file_list, _, _, _ = dataset
        self.seq_max_len = seq_max_len
        self.embed_dim = embed_dim
        n_stls = file_list[-1][4] + 1
        n_split = int(n_stls * 0.8)

        rng_state = torch.get_rng_state()
        perm_stl_indices = torch.randperm(n_stls, generator=torch.Generator().manual_seed(8008208820))
        torch.set_rng_state(rng_state)

        train_indices = perm_stl_indices[:n_split]
        val_indices = perm_stl_indices[n_split:]

        if split == "train":
            indices = train_indices
        elif split == "val":
            indices = val_indices
        elif split == "full":
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

        if args.test and args.seed != 1007:
            perm_stl_indices = torch.randperm(len(self.fs_list), generator=torch.Generator().manual_seed(args.seed))
            self.fs_list = [self.fs_list[idxidx] for idxidx in perm_stl_indices]

        if not args.concise:
            print("%s-len:%d" % (split, len(self.fs_list)))
        self.cache_tmp = None

    def len(self):
        return len(self.fs_list)

    def get(self, index):
        args = self.args
        demo_list, stl_data_list, simple_stl_list, real_stl_list, obj_list, file_list, stl_str_list, type_list, cache = self.dataset
        rec_i, stl_i, ego_i, sol_i, rel_stl_index, rel_ego_i, rel_sol_i = self.fs_list[index]

        sa_partial = torch.tensor([0])

        if (args.rand_aug_eval == False and args.rand_aug_graph) or (args.rand_aug_eval and self.split == "val") or args.test:
            the_simple_stl = rand_aug(simple_stl_list[stl_i], inplace=False)
        else:
            the_simple_stl = simple_stl_list[stl_i]

        if args.data_aug and self.split == "train":
            the_simple_stl = hard_rand_aug(simple_stl_list[stl_i], cfg={"tmax": args.horizon}, inplace=False)

        record = demo_list[rec_i]

        ego_state = torch.from_numpy(record["state"][ego_i * 8 + sol_i]).float()
        us = torch.from_numpy(record["us"][ego_i, sol_i]).float()
        traj = torch.from_numpy(record["trajs"][ego_i * 8 + sol_i]).float()
        sa_traj = torch.cat([traj[..., :-1, :], us], dim=-1)

        stl_embed = get_graph_stl_embed_from_tree(the_simple_stl, args, is_train=self.split == "train")

        index_tensor = torch.tensor(index)
        stl_i_tensor = torch.tensor(stl_i)
        stl_type_i_tensor = torch.tensor(record["stl_type_i"])

        # gnn encoder uses dummy placeholder for stl_str
        stl_str = torch.ones(1) * -1

        return index_tensor, stl_embed, ego_state, us, sa_traj, stl_str, sa_partial, stl_i_tensor, stl_type_i_tensor


def get_data_loader(dataset, seq_max_len, embed_dim, args):
    if args.same_train_val:
        train_dataset = GSTLDataset(dataset, "full", seq_max_len, embed_dim, args)
        val_dataset = GSTLDataset(dataset, "full", seq_max_len, embed_dim, args)
    else:
        train_dataset = GSTLDataset(dataset, "train", seq_max_len, embed_dim, args)
        val_dataset = GSTLDataset(dataset, "val", seq_max_len, embed_dim, args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not (args.no_shuffle),
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
