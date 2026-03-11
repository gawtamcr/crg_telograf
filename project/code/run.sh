python train.py \
        --flow \
        --first_sat_init \
        --skip_first_eval \
        --clip_max 50000 \
        --max_sol_clip 2 \
        --type_ratios 1 3 2 4 \
        -T g0128-075243_simple_gnn_F \
        --num_evals 128 \
        -b 1 \
        --test_muls 1024

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
