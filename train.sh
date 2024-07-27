python train.py \
  -b 1 \
  -e 2 \
  -l 0.001 \
  -r 500 \
  -n 'test_Varnet' \
  -t '/home/Data/train/' \
  -v '/home/Data/val/' \
  --cascade 12 \ 
  --aug_on \
  --aug_schedule exp \
  --aug_delay 1 \
  --aug_strength 0.5 \
  --aug_exp_decay 3.0 \