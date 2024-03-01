python test.py --data_dir=/mntcephfs/data/med/penet \
                 --ckpt_path=/mntcephfs/lab_data/wangcm/wzp/train_logs-main/PEMVCNet/best.pth.tar \
                 --results_dir=results \
                 --phase=test \
                 --name=PEMVCNet \
                 --dataset=pe \
                 --gpu_ids=0