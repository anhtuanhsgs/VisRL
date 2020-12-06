cat $0
python main.py --env 3D_1rgba_FixedMod \
--gpu-id 0 1 2 3 --workers 12 --valid-gpu 0 \
--data 3DVols \
--lr 1e-4 \
--num-actions 4 --color-step 16 \
--num-steps 4 --max-episode-length 4 \
--size 32 32 \
--feats 64 64 128 128 512 \
--save-period 100 --log-period 10 --train-log-period 100 \
--log-dir logs/Dec2020/ --save-model-dir logs/trained_models/Dec2020/ \
# --load logs/trained_models/Dec2020/n2RGB_FixedMod_ENet_800.dat \
\
