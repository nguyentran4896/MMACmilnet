from denoising_diffusion_pytorch import Trainer
from classifier_free_guidance import Unet, GaussianDiffusion


model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    num_classes = 5,
    cond_drop_prob = 0.5
)

diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    '/data_sda/zql/category/',
    train_batch_size = 24,
    train_lr = 8e-5,
    num_samples = 9,
    save_and_sample_every=10000,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False              # whether to calculate fid during training
)

trainer.train()

