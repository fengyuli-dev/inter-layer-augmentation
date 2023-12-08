# Command for interactive session:
srun --gres gpu:3090:1 -n 4 -t 48:00:00 -p kilian --mem 5G --pty /bin/bash
srun --gres gpu:a6000:1 -n 4 -t 48:00:00 -p kilian --mem 32G --pty /bin/bash
cp -r /home/ys732/share/datasets/amodal_Ithaca365 /scratch
# Command for observing gpu condition
watch -d -n 0.5 nvidia-smi
# jupyter-notebook
XDG_RUNTIME_DIR=/tmp/yw583 jupyter-notebook --ip=0.0.0.0 --port=8888
# small runs
srun --gres gpu:a6000:1 -n 4 -t 2:00:00 -p kilian --mem 2G --pty /bin/bash