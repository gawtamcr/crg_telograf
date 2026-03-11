## For testing
python -B train.py \
        --flow \
        --first_sat_init \
        --skip_first_eval \
        --clip_max 50000 \
        --max_sol_clip 2 \
        --type_ratios 1 3 2 4 \
        -T simple_gnn_F \
        --num_evals 128 \
        -b 1 \
        --max_viz 128 \
        --test_muls 1024 \
        --val_eval_only

## For testing with augmentation OOD
# python -B train.py \
#         --flow \
#         --first_sat_init \
#         --skip_first_eval \
#         --clip_max 50000 \
#         --max_sol_clip 2 \
#         --type_ratios 1 3 2 4 \
#         -T g0311-094109_simple_gnn_F \
#         --num_evals 128 \
#         -b 1 \
#         --max_viz 128 \
#         --test_muls 1024 \
#         --suffix _aug1 \
#         --aug_graph \
#         --max_aug 1 \
#         --val_eval_only

# ### For Training
# python -B train.py \
#         --flow \
#         --first_sat_init \
#         --skip_first_eval \
#         --clip_max 50000 \
#         --max_sol_clip 2 \
#         --type_ratios 1 3 2 4

# python train_gstl_v1.py \
#         --env simple \
#         --encoder gnn \
#         --flow \
#         --first_sat_init \
#         --skip_first_eval \
#         --clip_max 50000 \
#         --max_sol_clip 2 \
#         --type_ratios 1 3 2 4 \
#         -T g0128-075243_simple_gnn_F \
#         --fix \
#         --num_evals 128 \
#         -b 1 \
#         --test_muls 1024
