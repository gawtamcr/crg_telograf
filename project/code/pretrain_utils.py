import torch
import torch.nn.functional as F
import numpy as np
import utils
import wandb
from train_utils import parse_batch, get_encoding, mean_func, normalize_traj, set_model

def sst_encoder(encoder, tuple_data, train_loader, val_loader, args, device):
    demo_list, stl_data_list, simple_stl_list, real_stl_list, obj_list, file_list, stl_str_list, type_list, cache = tuple_data
    encoder = encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    eta = utils.EtaEstimator(start_iter=0, end_iter=args.epochs * len(train_loader))
    scheduler = utils.create_custom_lr_scheduler(optimizer, warmup_epochs=args.warmup_epochs, warmup_lr=args.warmup_lr, decay_epochs=args.decay_epochs, decay_lr=args.decay_lr, decay_mode=args.decay_mode)

    if args.predict_score:
        for epi in range(args.epochs):
            md = utils.MeterDict()
            if not args.concise:
                print("Epochs[%03d/%03d] lr:%.7f" % (epi, args.epochs, optimizer.param_groups[0]['lr']))
            for mode, sel_loader in [("train", train_loader), ("val", val_loader)]:
                set_model(encoder, mode)
                all_y_preds1 = []
                all_y_preds2 = []
                all_y_gt = []
                stl_type_gt = []
                for bi, batch in enumerate(sel_loader):
                    eta.update()
                    index_tensors, stl_embeds, ego_states, us, sa_trajectories, stl_str, sa_par, stl_i_tensors, stl_type_i_tensors = parse_batch(batch, device)

                    trajs2d = sa_trajectories[:, :, :2].reshape(sa_trajectories.shape[0], -1)
                    traj_embed = encoder.predict_score_head(trajs2d)
                    traj_embed = traj_embed / torch.clip(torch.norm(traj_embed, dim=-1, keepdim=True), min=1e-4)

                    stl_embed = encoder(None, stl_embeds)
                    stl_embed = stl_embed / torch.clip(torch.norm(stl_embed, dim=-1, keepdim=True), min=1e-4)
                    logits_pred = torch.mm(stl_embed, traj_embed.t())
                    batch_size = traj_embed.size(0)

                    y_gt = labels = torch.arange(batch_size, device=traj_embed.device)
                    loss1 = F.cross_entropy(logits_pred, labels)
                    loss2 = F.cross_entropy(logits_pred.T, labels)
                    loss = (loss1 + loss2) / 2

                    if mode == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    y_pred1 = torch.argmax(logits_pred, dim=1)
                    y_pred2 = torch.argmax(logits_pred, dim=0)
                    acc1 = torch.mean((y_pred1 == labels).float())
                    acc2 = torch.mean((y_pred2 == labels).float())
                    all_y_preds1.append(y_pred1)
                    all_y_preds2.append(y_pred2)
                    all_y_gt.append(y_gt)
                    stl_type_gt.append(stl_type_i_tensors)

                    md.update("%s_loss" % (mode), loss.item())
                    md.update("%s_acc" % (mode), (acc1.item() + acc2.item()) / 2)
                    # Logging logic...
                    
                # Evaluation Logic...
                print(f"{mode} epoch {epi} finished.")
                
            scheduler.step()
            utils.save_model_freq_last(encoder.state_dict(), args.model_dir, epi, args.save_freq, args.epochs)

    if args.with_predict_head:
        CE_loss = torch.nn.CrossEntropyLoss()
        for epi in range(args.epochs):
            md = utils.MeterDict()
            if not args.concise:
                print("Epochs[%03d/%03d] lr:%.7f" % (epi, args.epochs, optimizer.param_groups[0]['lr']))
            for mode, sel_loader in [("train", train_loader), ("val", val_loader)]:
                set_model(encoder, mode)
                all_logits = []
                all_y_preds = []
                all_y_gt = []
                for bi, batch in enumerate(sel_loader):
                    eta.update()
                    index_tensors, stl_embeds, ego_states, us, sa_trajectories, stl_str, sa_par, stl_i_tensors, stl_type_i_tensors = parse_batch(batch, device)
                    y_gt = stl_type_i_tensors
                    embedding = encoder(None, stl_embeds)
                    logits_pred = encoder.predict(embedding)
                    y_pred = torch.argmax(logits_pred, dim=-1)
                    loss = CE_loss(logits_pred, y_gt)

                    if mode == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    acc = torch.mean((y_pred == y_gt).float())
                    # Accumulate stats...
                    md.update("%s_loss" % (mode), loss.item())
                    md.update("%s_acc" % (mode), acc.item())

                # Final epoch stats...
                print(f"{mode} epoch {epi} finished.")

            scheduler.step()
            utils.save_model_freq_last(encoder.state_dict(), args.model_dir, epi, args.save_freq, args.epochs)
    return

def train_score_predictor(score_predictor, tuple_data, train_loader, val_loader, stat_mean, stat_std, args, device):
    demo_list, stl_data_list, simple_stl_list, real_stl_list, obj_list, file_list, stl_str_list, type_list, cache = tuple_data
    score_predictor = score_predictor.to(device)
    optimizer = torch.optim.Adam(score_predictor.parameters(), lr=args.lr)
    eta = utils.EtaEstimator(start_iter=0, end_iter=args.epochs * len(train_loader))
    scheduler = utils.create_custom_lr_scheduler(optimizer, warmup_epochs=args.warmup_epochs, warmup_lr=args.warmup_lr, decay_epochs=args.decay_epochs, decay_lr=args.decay_lr, decay_mode=args.decay_mode)
    stl_acc_log_list = []

    for epi in range(args.epochs):
        md = utils.MeterDict()
        MINI_BATCH_N = 8
        if epi % 5 == 0 or epi == args.epochs - 1:
            stl_acc_log_list.append({"epoch": epi, "train": 0, "val": 0})
        if not args.concise:
            print("Epochs[%03d/%03d] lr:%.7f" % (epi, args.epochs, optimizer.param_groups[0]['lr']))
        for mode, sel_loader in [("train", train_loader), ("val", val_loader)]:
            set_model(score_predictor, mode)
            acc_d = {0: [], 1: [], 2: [], 3: []}

            for bi, batch in enumerate(sel_loader):
                eta.update()
                index_tensors, stl_embeds, ego_states, us, sa_trajectories, stl_str, sa_par, stl_i_tensors, stl_type_i_tensors = parse_batch(batch, device)
                BS = ego_states.shape[0]

                noised_trajs = sa_trajectories + torch.randn_like(sa_trajectories)
                total_trajs = torch.stack([sa_trajectories, noised_trajs], dim=0)  # (2, BS, NT, K)

                scores_gt_list = []
                for iii in range(MINI_BATCH_N):
                    rec_i, stl_i, ego_i, sol_i, rel_stl_index, rel_ego_i, rel_sol_i = sel_loader.dataset.fs_list[index_tensors[iii].item()]
                    real_stl = real_stl_list[stl_i]
                    scores_gt = real_stl(total_trajs[:, iii], args.smoothing_factor)[:, 0]
                    scores_gt_list.append(scores_gt)
                scores_gt_list = torch.stack(scores_gt_list, dim=1)  # (2, MINIBATCH)
                y_gt = torch.clip(scores_gt_list, -1, 1)

                total_trajs = total_trajs.reshape(2 * BS, total_trajs.shape[-2], total_trajs.shape[-1])
                normalized_sa_trajs = normalize_traj(total_trajs, stat_mean=stat_mean, stat_std=stat_std)
                normalized_sa_trajs_flat = normalized_sa_trajs.reshape(2 * BS, -1)

                stl_feat = get_encoding(score_predictor.encoder, None, stl_embeds, stl_str, args)
                y_pred = score_predictor.dual_forward(None, stl_embeds, normalized_sa_trajs_flat, MINI_BATCH_N, stl_feat=stl_feat)

                loss = torch.mean(torch.square(y_pred - y_gt))
                if mode == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                acc_all = (torch.sign(y_pred) == torch.sign(y_gt)).float()
                acc = torch.mean(acc_all)
                md.update("%s_loss" % (mode), loss.item())
                md.update("%s_acc" % (mode), acc.item())

        scheduler.step()
        utils.save_model_freq_last(score_predictor.state_dict(), args.model_dir, epi, args.save_freq, args.epochs)
    return