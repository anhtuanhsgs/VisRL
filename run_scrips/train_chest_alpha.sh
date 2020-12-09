cat $0
python main.py --env chest_alpha \
--gpu-id 0 1 2 3 --workers 8 --valid-gpu 0 \
--data 3DChest \
--model Net3D \
--lr 1e-4 \
--num-actions 7 --color-step 16 \
--alpha-only \
--num-steps 4 --max-episode-length 4 \
--size 128 128 128 \
--feats 64 64 128 512 1024 \
--save-period 100 --log-period 10 --train-log-period 100 \
--log-dir logs/Dec2020/ --save-model-dir logs/trained_models/Dec2020/ \
# --load logs/trained_models/Dec2020/3D_1rgba_LimitedMod/7700.dat \
\
