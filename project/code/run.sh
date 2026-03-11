## Training
# python -B train.py --config configs/train.yaml

## Testing
python -B train.py --config configs/test.yaml -T simple_gnn_F

## Testing with augmentation OOD  (edit configs/test_aug.yaml as needed)
# python -B train.py --config configs/test_aug.yaml -T g0311-094109_simple_gnn_F
