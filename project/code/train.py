import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import utils
import tqdm
import wandb

from z_diffuser import GaussianDiffusion, GaussianFlow, TemporalUnet, MockNet, MLPNet, GaussianVAE
from torch_geometric.loader import DataLoader
import generate_scene_v1

from matplotlib.patches import Circle
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from z_models import GCN, ScorePredictor
from dataset import load_dataset, get_data_loader, get_graph_stl_embed_from_tree
from train_utils import get_encoding, normalize_traj, denorm_traj, parse_batch, get_denoising_results, set_model, mean_func
from pretrain_utils import sst_encoder, train_score_predictor
from vis_utils import visualize_plan

from os.path import join as ospj


def main():
    utils.setup_exp_and_logger(args, test=args.test, dryrun=args.dryrun)

    if not args.dryrun:
        wandb.init(entity="gawtamcr-kth", project="diffusion",  name=args.exp_name, config=vars(args))

    print("Env:%s  use data %s ..." % (args.env, args.data_path))
    tuple_data = load_dataset(args)
    demo_list, stl_data_list, simple_stl_list, real_stl_list, obj_list, file_list, stl_str_list, type_list, cache = tuple_data
    seq_max_len = np.max([len(xxx) for xxx in stl_str_list])

    # gnn encoder: input is graph node features (8-dim or 14-dim for hashi)
    input_dim = 14 if args.hashi_gnn else 8

    train_loader, val_loader = get_data_loader(tuple_data, seq_max_len, input_dim, args)
    print("seq_max_len", seq_max_len)

    # Statistics for simple env (meanstd normalization)
    x_min, x_max, y_min, y_max = -5, 5, -5, 5
    stat_mean = torch.Tensor([0, 0, 0, 0]).to(device)
    stat_std = torch.Tensor([5, 5, 1, 1]).to(device)

    ego_state_dim = args.observation_dim
    encoder = GCN(input_dim, ego_state_dim, args).to(device)

    if args.cls_guidance:
        encoder_extra = GCN(8, ego_state_dim, args).to(device)

    if args.sst:
        sst_encoder(encoder, tuple_data, train_loader, val_loader, args, device)
        return

    if args.mock_model:
        model = MockNet(args.horizon, args.data_dim,
                        cond_dim=args.condition_dim + ego_state_dim,
                        dim=args.tconv_dim,
                        dim_mults=args.dim_mults,
                        attention=args.attention)
    elif args.mlp:
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

    ## Setting model class
    if args.flow:
        gen_model_class = GaussianFlow
    elif args.vae:
        gen_model_class = GaussianVAE
    else:
        gen_model_class = GaussianDiffusion

    diffuser = gen_model_class(
        model=model,
        horizon=args.horizon,
        observation_dim=args.data_dim - args.action_dim,
        action_dim=args.action_dim,
        n_timesteps=args.n_timesteps,
        loss_type=args.loss_type,
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
            res = encoder.load_state_dict(enc_checkpoints, strict=False)
            print(res.missing_keys)
        else:
            diffuser.load_state_dict(checkpoint)
    diffuser = diffuser.to(device)

    if args.train_encoder:
        trainable_names = [name for name, param in diffuser.named_parameters() if not name.startswith('model.')]
        if args.train_unet_partial:
            trainable_names += [name for name, param in diffuser.named_parameters()
                                 if name.startswith('model.cond_mlp') or (name.startswith("model.time_mlp") == False and "time_mlp" in name)]
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
            {'params': trainable_params, 'lr': args.lr},
            {'params': finetune_params, 'lr': args.lr * 0.1}
        ])
    else:
        optimizer = torch.optim.Adam(diffuser.parameters(), lr=args.lr)

    scheduler = utils.create_custom_lr_scheduler(optimizer, warmup_epochs=args.warmup_epochs, warmup_lr=args.warmup_lr, decay_epochs=args.decay_epochs, decay_lr=args.decay_lr, decay_mode=args.decay_mode)

    if args.set_detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    if args.train_classifier or args.cls_guidance:
        if args.cls_guidance:
            score_predictor = ScorePredictor(encoder_extra, args.condition_dim, args.horizon, args.data_dim, args).to(device)
        else:
            score_predictor = ScorePredictor(encoder, args.condition_dim, args.horizon, args.data_dim, args).to(device)
            train_score_predictor(score_predictor, tuple_data, train_loader, val_loader, stat_mean, stat_std, args, device)
            return

    if args.cls_path is not None and (args.train_classifier or args.cls_guidance):
        checkpoint_cls = torch.load(os.path.join(utils.get_exp_dir(), utils.smart_path(args.cls_path)))
        score_predictor.load_state_dict(checkpoint_cls)

    def loss_func(test_x, test_stl):
        score = test_stl(test_x, args.smoothing_factor)[:, 0]
        loss = torch.mean(torch.nn.ReLU()(0.5 - score))
        return loss, {}

    # -------------------------------------------------------------------------
    # TEST / EVAL
    # -------------------------------------------------------------------------
    if args.test:
        md = utils.MeterDict()
        diffuser.eval()
        stl_acc_log_list = [{"epoch": 0, "train": 0, "val": 0}]
        eval_stat = {"train": {"stl": {}}, "val": {"stl": {}}}
        res_d = {"train": [], "val": []}
        res_d["meta"] = {"env": args.env, "encoder": args.encoder, "args": args, "exp_dir_full": args.exp_dir_full}
        viz_cnt = {"train": 0, "val": 0}
        eta = utils.EtaEstimator(0, args.num_evals * 2)

        for mode, sel_loader in [("train", train_loader), ("val", val_loader)]:
            eval_scores = []
            t_avg_list = []
            eval_scores_d = {}
            eval_split = mode
            if args.val_eval_only and mode == "train":
                continue
            if args.train_eval_only and mode == "val":
                continue

            for bi, batch in enumerate(sel_loader):
                if bi >= args.num_evals:
                    break
                eta.update()
                RES = {}
                res_d[mode].append(RES)

                index_tensors, stl_embeds, ego_states, us, sa_trajectories, stl_str, sa_par, stl_i_tensors, stl_type_i_tensors = parse_batch(batch, device)

                compute_t1 = time.time()
                bs = sa_trajectories.shape[0]

                # learning-based inference
                if args.cls_guidance:
                    stl_embeds, stl_embeds_gnn = stl_embeds
                conditions = get_encoding(encoder, ego_states, stl_embeds, stl_str, args)
                if args.test_muls is not None:
                    conditions = conditions[:, None].repeat(1, args.test_muls, 1).reshape(bs * args.test_muls, conditions.shape[-1])

                if args.guidance or args.cls_guidance:
                    tmp_real_stl_list = []
                    for iii in range(ego_states.shape[0]):
                        rec_i, stl_i, ego_i, sol_i, rel_stl_index, rel_ego_i, rel_sol_i = sel_loader.dataset.fs_list[index_tensors[iii].item()]
                        tmp_real_stl_list.append(real_stl_list[stl_i])

                    def _norm(t):
                        return normalize_traj(t, stat_mean, stat_std)

                    def _denorm(t):
                        return denorm_traj(t, stat_mean, stat_std)

                    guidance_data = {"args": args, "denorm": _denorm, "norm_func": _norm,
                                     "loss_func": loss_func, "real_stl_list": tmp_real_stl_list}
                    if args.cls_guidance:
                        guidance_data["score_predictor"] = score_predictor
                        guidance_data["stl_embeds"] = stl_embeds
                        guidance_data["stl_embeds_gnn"] = stl_embeds_gnn
                else:
                    guidance_data = None

                diffused_trajs = get_denoising_results(diffuser, conditions, stat_mean, stat_std, args, guidance_data=guidance_data)

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
                    eval_stat[eval_split]["stl"][stl_i] += 1

                    max_stl_i = 0
                    if args.test_muls is not None:
                        score = real_stl(diffused_trajs[iii * args.test_muls:(iii + 1) * args.test_muls], args.smoothing_factor)[:, 0]
                        score, max_stl_i = torch.max(score, dim=0)
                        score = score[None]
                    else:
                        score = real_stl(diffused_trajs[iii:iii + 1], args.smoothing_factor)[:, 0]

                    eval_scores.append(int(score.item() > 0))
                    if stl_type not in eval_scores_d:
                        eval_scores_d[stl_type] = []
                    eval_scores_d[stl_type].append(int(score.item() > 0))

                    RES["scores"].append(score.item())
                    RES["acc"].append(int(score.item() > 0))
                    if args.test_muls is not None:
                        best_traj = diffused_trajs[iii * args.test_muls + max_stl_i]
                    else:
                        best_traj = diffused_trajs[iii]
                    RES['trajs'].append(utils.to_np(best_traj))

                RES['trajs'] = np.stack(RES['trajs'], axis=0)

                print("%-5s [%03d/%03d] acc:%d (%.3f) runtime:%.3f(%.3f)   dt:%s  Elapsed:%s  ETA:%s" % (
                    mode.upper(), bi, args.num_evals, RES["acc"][-1], np.mean(eval_scores), RES["t"], np.mean(t_avg_list),
                    eta.interval_str(), eta.elapsed_str(), eta.eta_str()
                ))

                # Visualization
                diffused_trajs_np = utils.to_np(diffused_trajs)
                gt_trajs_np = utils.to_np(sa_trajectories.reshape(sa_trajectories.shape[0], args.horizon, args.observation_dim + args.action_dim)[:, :, :2])
                n_viz_trajs_max = args.n_viz_trajs_max

                for mini_i in range(index_tensors.shape[0]):
                    if viz_cnt[mode] >= args.max_viz:
                        continue
                    viz_cnt[mode] += 1

                    rec_i, stl_i, ego_i, sol_i, rel_stl_index, rel_ego_i, rel_sol_i = sel_loader.dataset.fs_list[index_tensors[mini_i].item()]
                    simp_stl = simple_stl_list[stl_i]
                    real_stl = real_stl_list[stl_i]
                    objects_d = obj_list[stl_i]

                    visualize_plan(
                        mode=mode, bi=bi, mini_i=mini_i,
                        gt_trajs_np=gt_trajs_np[mini_i],
                        diffused_trajs_np=diffused_trajs_np[mini_i * (args.test_muls or 1): (mini_i + 1) * (args.test_muls or 1)],
                        res_trajs=RES['trajs'][mini_i] if args.test_muls is not None else None,
                        simple_stl=simp_stl, objects_d=objects_d, args=args,
                        x_lims=(x_min, x_max), y_lims=(y_min, y_max)
                    )

            stl_acc_log_list[-1][eval_split] = np.mean(eval_scores)
            for stl_type in sorted(eval_scores_d.keys()):
                stl_acc_log_list[-1][eval_split + "_%d" % stl_type] = np.mean(eval_scores_d[stl_type])
            print("%s STL Acc:%.3f" % (eval_split, np.mean(eval_scores)))
            if bi % 20 == 0:
                np.savez("%s/stl_acc.npz" % (args.exp_dir_full), data=stl_acc_log_list)
                np.savez("%s/results.npz" % (args.exp_dir_full), data=res_d)

        train_acc_str = " ".join(["%s:%.4f" % (stl_type[5:], stl_acc_log_list[-1][stl_type]) for stl_type in stl_acc_log_list[-1] if "train" in stl_type])
        val_acc_str = " ".join(["%s:%.4f" % (stl_type[5:], stl_acc_log_list[-1][stl_type]) for stl_type in stl_acc_log_list[-1] if "val" in stl_type])
        print("STAT train:%d stls, %d trajs (%s) | val:%d stls, %d trajs (%s)" % (
            len(eval_stat["train"]["stl"]), sum(eval_stat["train"]["stl"].values()), train_acc_str,
            len(eval_stat["val"]["stl"]), sum(eval_stat["val"]["stl"].values()), val_acc_str,
        ))
        np.savez("%s/stl_acc.npz" % (args.exp_dir_full), data=stl_acc_log_list)
        np.savez("%s/results.npz" % (args.exp_dir_full), data=res_d)
        return

    # -------------------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------------------
    else:
        stl_acc_log_list = []
        train_step = 0
        start_epi = 0
        CE_loss = torch.nn.CrossEntropyLoss()

        if args.rebase is not None:
            start_epi = args.rebase
            train_step = start_epi * len(train_loader)
        eta = utils.EtaEstimator(start_iter=start_epi, end_iter=args.epochs * len(train_loader))

        for epi in range(start_epi, args.epochs):
            md = utils.MeterDict()
            all_logits = []
            all_y_preds = []
            all_y_gt = []
            if not args.concise:
                print("Epochs[%03d/%03d] lr:%.7f" % (epi, args.epochs, optimizer.param_groups[0]['lr']))
            diffuser.train()

            for bi, batch in enumerate(train_loader):
                eta.update()
                if args.debug:
                    continue

                index_tensors, stl_embeds, ego_states, us, sa_trajectories, stl_str, sa_par, stl_i_tensors, stl_type_i_tensors = parse_batch(batch, device)

                normalized_sa_trajs = normalize_traj(sa_trajectories, stat_mean=stat_mean, stat_std=stat_std)

                batch_size = normalized_sa_trajs.shape[0]

                conditions = get_encoding(encoder, ego_states, stl_embeds, stl_str, args)

                t = torch.randint(0, args.n_timesteps, (batch_size,), device=device).long()
                noise = torch.randn_like(normalized_sa_trajs)
                x_noisy = diffuser.q_sample(x_start=normalized_sa_trajs, t=t, noise=noise)
                x_noisy = x_noisy.reshape(batch_size, args.horizon, args.data_dim)
                noise = noise.reshape(batch_size, args.horizon, args.data_dim)

                x_recon = diffuser.model(x_noisy, conditions, t)

                # Set target for loss calculation
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
                    group_labels = stl_i_tensors.unsqueeze(1)
                    pos_mask = group_labels == group_labels.t()
                    embedding = encoder(None, stl_embeds)
                    embedding = embedding / torch.clip(torch.norm(embedding, dim=-1, keepdim=True), min=1e-4)
                    logits_pred = torch.mm(embedding, embedding.t())
                    log_probs_x_to_y = F.log_softmax(logits_pred, dim=1)
                    log_probs_y_to_x = F.log_softmax(logits_pred.t(), dim=1)
                    positive_log_probs_x_to_y = log_probs_x_to_y[pos_mask]
                    positive_log_probs_y_to_x = log_probs_y_to_x[pos_mask]
                    loss1 = -positive_log_probs_x_to_y.mean()
                    loss2 = -positive_log_probs_y_to_x.mean()
                    loss = loss + (loss1 + loss2) / 2

                add_str = ""
                if args.with_predict_head:
                    embedding = encoder(None, stl_embeds)
                    y_gt = stl_type_i_tensors
                    logits_pred = encoder.predict(embedding)
                    y_pred = torch.argmax(logits_pred, dim=-1)
                    ce_loss = CE_loss(logits_pred, y_gt)
                    old_loss = loss
                    loss = old_loss + ce_loss
                    sst_acc = torch.mean((y_pred == y_gt).float())
                    all_logits.append(logits_pred)
                    all_y_preds.append(y_pred)
                    all_y_gt.append(y_gt)
                    md.update("acc", sst_acc.detach().item())
                    md.update("old_loss", old_loss.detach().item())
                    md.update("ce_loss", ce_loss.detach().item())
                    add_str = " | old_loss:%.3f(%.3f) ce_loss:%.3f(%.3f) acc:%.3f(%.3f) |" % (
                        md["old_loss"], md("old_loss"), md["ce_loss"], md("ce_loss"),
                        md["acc"], md("acc")
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                md.update("loss", loss.detach().item())

                if bi % args.print_freq == 0:
                    wandb.log({"train/loss": loss.detach().item(), "epoch": epi})
                    if args.with_predict_head:
                        wandb.log({"train/acc": md("acc"), "train/ce_loss": md("ce_loss")})
                    print("Epochs[%03d/%03d] (%04d/%04d) loss:%.3f(%.3f) %s dt:%s Elapsed:%s ETA:%s" % (
                        epi, args.epochs, bi, len(train_loader),
                        md["loss"], md("loss"), add_str, eta.interval_str(), eta.elapsed_str(), eta.eta_str()))

            scheduler.step()
            utils.save_model_freq_last(diffuser.state_dict(), args.model_dir, epi, args.save_freq, args.epochs)

            # Evaluate
            if (epi % args.eval_freq == 0 or epi == args.epochs - 1) and args.debug == False and (args.skip_first_eval == False or epi != 0):
                diffuser.eval()
                stl_acc_log_list.append({"epoch": epi, "train": 0, "val": 0})
                eval_stat = {"train": {"stl": {}}, "val": {"stl": {}}}
                for eval_split, sel_loader in [("train", train_loader), ("val", val_loader)]:
                    eval_scores = []
                    eval_scores_d = {}
                    for bi, batch in enumerate(sel_loader):
                        if bi >= args.num_evals:
                            continue
                        index_tensors, stl_embeds, ego_states, us, sa_trajectories, stl_str, sa_par, stl_i_tensors, stl_type_i_tensors = parse_batch(batch, device)
                        conditions = get_encoding(encoder, ego_states, stl_embeds, stl_str, args)
                        diffused_trajs = get_denoising_results(diffuser, conditions, stat_mean, stat_std, args)

                        for iii in range(diffused_trajs.shape[0]):
                            rec_i, stl_i, ego_i, sol_i, rel_stl_index, rel_ego_i, rel_sol_i = sel_loader.dataset.fs_list[index_tensors[iii].item()]
                            real_stl = real_stl_list[stl_i]
                            stl_type = type_list[stl_i]

                            if stl_i not in eval_stat[eval_split]["stl"]:
                                eval_stat[eval_split]["stl"][stl_i] = 0
                            eval_stat[eval_split]["stl"][stl_i] += 1

                            score = real_stl(diffused_trajs[iii:iii + 1], args.smoothing_factor)[:, 0]
                            eval_scores.append(int(score.item() > 0))
                            if stl_type not in eval_scores_d:
                                eval_scores_d[stl_type] = []
                            eval_scores_d[stl_type].append(int(score.item() > 0))

                    stl_acc_log_list[-1][eval_split] = np.mean(eval_scores)
                    for stl_type in sorted(eval_scores_d.keys()):
                        stl_acc_log_list[-1][eval_split + "_%d" % stl_type] = np.mean(eval_scores_d[stl_type])
                    print("%s STL Acc:%.3f" % (eval_split, np.mean(eval_scores)))
                    if not args.test:
                        wandb.log({f"eval/{eval_split}_acc": np.mean(eval_scores), "epoch": epi})

                train_acc_str = " ".join(["%s:%.4f" % (stl_type[5:], stl_acc_log_list[-1][stl_type]) for stl_type in stl_acc_log_list[-1] if "train" in stl_type])
                val_acc_str = " ".join(["%s:%.4f" % (stl_type[5:], stl_acc_log_list[-1][stl_type]) for stl_type in stl_acc_log_list[-1] if "val" in stl_type])
                print("STAT train:%d stls, %d trajs (%s) | val:%d stls, %d trajs (%s)" % (
                    len(eval_stat["train"]["stl"]), sum(eval_stat["train"]["stl"].values()), train_acc_str,
                    len(eval_stat["val"]["stl"]), sum(eval_stat["val"]["stl"].values()), val_acc_str,
                ))
                np.savez("%s/stl_acc.npz" % (args.exp_dir_full), data=stl_acc_log_list)
                diffuser.train()

            # Visualization
            if (epi % args.viz_freq == 0 or epi == args.epochs - 1) and (args.skip_first_eval == False or epi != 0):
                sel_loader = train_loader if args.debug else val_loader
                max_b = 100 if args.debug else 1
                for bi, batch in enumerate(sel_loader):
                    if bi >= max_b:
                        continue
                    index_tensors, stl_embeds, ego_states, us, sa_trajectories, stl_str, sa_par, stl_i_tensors, stl_type_i_tensors = parse_batch(batch, device)
                    conditions = get_encoding(encoder, ego_states, stl_embeds, stl_str, args)

                    n_viz = 100
                    diffused_trajs = get_denoising_results(diffuser, conditions[0:1].repeat(n_viz, 1), stat_mean, stat_std, args)
                    diffused_trajs_np = utils.to_np(diffused_trajs)
                    gt_trajs_np = utils.to_np(sa_trajectories.reshape(sa_trajectories.shape[0], args.horizon, args.observation_dim + args.action_dim)[0, :, :2])

                    rec_i, stl_i, ego_i, sol_i, rel_stl_index, rel_ego_i, rel_sol_i = sel_loader.dataset.fs_list[index_tensors[0].item()]
                    simp_stl = simple_stl_list[stl_i]
                    objects_d = obj_list[stl_i]

                    visualize_plan(
                        mode="train", bi=bi, mini_i=0,
                        gt_trajs_np=gt_trajs_np,
                        diffused_trajs_np=diffused_trajs_np,
                        simple_stl=simp_stl, objects_d=objects_d, args=args,
                        x_lims=(x_min, x_max), y_lims=(y_min, y_max), step=epi
                    )

    return


if __name__ == "__main__":
    import yaml

    # Usage:
    #   python train.py --config configs/train.yaml          # training
    #   python train.py --config configs/test.yaml -T <exp>  # testing
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="YAML config file (stacked on configs/default.yaml)")
    parser.add_argument("-T", "--test_path", type=str, default=None, help="Pretrained model path; implies test mode")
    cli = parser.parse_args()

    _here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(_here, "configs", "default.yaml")) as _f:
        cfg = yaml.safe_load(_f) or {}

    if cli.config is not None:
        with open(cli.config) as _f:
            cfg.update(yaml.safe_load(_f) or {})

    if cli.test_path is not None:
        cfg["test_path"] = cli.test_path

    args = argparse.Namespace(**cfg)

    # Fixed: simple env, gnn encoder
    args.env = "simple"
    args.encoder = "gnn"
    args.observation_dim = 2
    args.action_dim = 2
    args.data_dim = args.observation_dim + args.action_dim

    if args.cpu:
        device = "cpu"
    else:
        device = "cuda:0"

    if args.exp_name is None:
        args.exp_name = "simple_gnn_F_crg_big_train"
        if args.seed != 1007:
            args.exp_name += "_%d" % (args.seed)
        print("setup exp name", args.exp_name)

    if "QTEST" in args.exp_name:
        args.concise = True

    if args.test_path is not None:
        args.test = True
        args.net_pretrained_path = args.test_path

    if args.post_residual:
        args.residual = True

    args.clip_denoised = True
    args.attention = True
    args.cached = True
    args.rand_aug_eval = True

    t1 = time.time()
    main()
    t2 = time.time()
    print("Finished in %.3f seconds" % (t2 - t1))