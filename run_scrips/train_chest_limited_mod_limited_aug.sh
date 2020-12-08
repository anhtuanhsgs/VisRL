cat $0
python main.py --env chest_1rgba_LimitedModLimitedAug \
--gpu-id 0 1 2 3 --workers 12 --valid-gpu 0 \
--data 3DChest \
--lr 1e-4 \
--num-actions 4 --color-step 16 \
--num-steps 3 --max-episode-length 3 \
--size 128 128 \
--feats 64 64 128 512 1024 \
--save-period 100 --log-period 10 --train-log-period 100 \
--log-dir logs/Dec2020/ --save-model-dir logs/trained_models/Dec2020/ \
# --load logs/trained_models/Dec2020/3D_1rgba_LimitedMod/7700.dat \
\
