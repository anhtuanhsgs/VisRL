cat $0
python main.py --env n2RGB_FullMod \
--gpu-id 0 1 2 3 --workers 8 --valid-gpu 0 \
--data Random \
--lr 1e-4 --num-actions 8 \
--num-steps 12 --max-episode-length 12 \
--size 32 32 \
--feats 64 64 128 128 512 \
--save-period 100 --log-period 5 --train-log-period 100 \
--log-dir logs/Dec2020/ --save-model-dir logs/trained_models/Dec2020/ \
--load logs/trained_models/Dec2020/n2RGB_FixedMod_ENet_800.dat \
\
