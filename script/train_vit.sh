# cifar10 with warmup
# python trainer.py --model vit --dataset cifar10 --batch_size 32 --lr_schedule cosine_warmup --warmup_epochs 10 --latent_dim 256

# cifar100
python trainer.py --model vit --dataset cifar100 --batch_size 128 --lr_schedule cosine_warmup --warmup_epochs 10 --latent_dim 1024
