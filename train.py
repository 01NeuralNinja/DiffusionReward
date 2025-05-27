import os
from argparse import ArgumentParser
import random
import lpips
import torch
import torchvision
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from diffbir.model import ControlLDM, SwinIR, Diffusion
from diffbir.utils.common import instantiate_from_config, to
from diffbir.sampler import DDIMSampler
from utils_refl import DWTLoss, PerceptualLoss, hps_loss_ru, setup_reward_model


def main(args) -> None:
    # Setup gradient accumulation
    gradient_accumulation_steps = getattr(args, "gradient_accumulation_steps", 4)

    # Load configuration
    cfg = OmegaConf.load(args.config)

    # Setup wandb logging
    accelerator = Accelerator(
        split_batches=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with=cfg.train.log_record
    )
    set_seed(231, device_specific=True)
    accelerator.init_trackers(
        "diffbir_ReFL",  # Project name
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    device = accelerator.device

    # Setup experiment folder
    if accelerator.is_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")
        print(f"Using gradient accumulation with {gradient_accumulation_steps} steps")

    # Create DiffBIR model
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)

    # Load pretrained Stable Diffusion weights
    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)
    if accelerator.is_main_process:
        print(f"Strictly load pretrained SD weight from {cfg.train.sd_path}\n"
              f"Unused weights: {unused}\n"
              f"Missing weights: {missing}")

    # Load ControlNet weights
    control_weight = torch.load(cfg.train.controlnet_path, map_location="cpu")
    unused, missing = cldm.load_controlnet_from_ckpt(control_weight)
    if accelerator.is_main_process:
        print(f"Strictly load ControlNet weight from {cfg.train.controlnet_path}\n"
              f"Unused weights: {unused}\n"
              f"Missing weights: {missing}")

    # Setup SwinIR model
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    sd = torch.load(cfg.train.swinir_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in sd.items()
    }
    swinir.load_state_dict(sd, strict=True)
    for p in swinir.parameters():
        p.requires_grad = False
    if accelerator.is_main_process:
        print(f"Load SwinIR from {cfg.train.swinir_path}")

    # Setup diffusion model
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)

    # Setup optimizer
    opt = torch.optim.AdamW(
        [p for p in cldm.controlnet.parameters() if p.requires_grad],
        lr=cfg.train.learning_rate
    )

    # Setup data loader
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    if accelerator.is_main_process:
        print(f"Dataset contains {len(dataset):,} images")

    # Setup batch transform
    batch_transform = instantiate_from_config(cfg.batch_transform)

    # Setup precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Setup loss functions
    dwt_loss = DWTLoss(
        loss_weight=cfg.reward.dwt_loss_weight,
        lowq=True,
        highq=False
    ).to(accelerator.device, dtype=torch.float32)

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # Setup Face Reward Model (FRM)
    reward_model, tokenizer, normalize, target_size = setup_reward_model(cfg, weight_dtype, accelerator)

    # FRM update training configuration
    update_freq = getattr(cfg.reward, "update_freq", 10)
    reward_learning_rate = getattr(cfg.reward, "learning_rate_reward", 2e-6)

    # Setup optimizer for FRM
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(reward_model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    optimizer_reward = torch.optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": 0.35},
        ],
        lr=reward_learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
    )

    lr_scheduler_reward = get_scheduler(
        "constant",
        optimizer=optimizer_reward,
        num_warmup_steps=500 * accelerator.num_processes,
        num_training_steps=cfg.train.train_steps * accelerator.num_processes,
    )

    # Prepare models for training
    cldm.train().to(device)
    swinir.eval().to(device)
    diffusion.to(device)

    # Prepare models for training (including FRM optimizer)
    cldm, opt, loader, reward_model, net_lpips, optimizer_reward, lr_scheduler_reward = accelerator.prepare(
        cldm, opt, loader, reward_model, net_lpips, optimizer_reward, lr_scheduler_reward
    )

    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    noise_aug_timestep = cfg.train.noise_aug_timestep

    # Variables for monitoring/logging purposes
    global_step = 0
    max_steps = cfg.train.train_steps
    epoch = 0

    # Setup sampler
    sampler = DDIMSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False, eta=1.0)

    if accelerator.is_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")

    # Training loop
    while global_step < max_steps:
        pbar = tqdm(
            iterable=None,
            disable=not accelerator.is_main_process,
            unit="batch",
            total=len(loader),
        )

        for batch in loader:
            # Use accelerator.accumulate context manager for training code
            m_acc = [cldm, reward_model]
            with accelerator.accumulate(*m_acc):
                # Set reward model to eval mode to avoid updating internal state
                reward_model.eval()
                for param in reward_model.parameters():
                    param.requires_grad = False
                cldm.train()

                # Prepare batch data
                to(batch, device)
                batch = batch_transform(batch)
                gt, lq, prompt, caption = batch
                gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()
                lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()

                # Forward pass preparation
                with torch.no_grad():
                    z_0 = pure_cldm.vae_encode(gt)
                    clean = swinir(lq)
                    cond = pure_cldm.prepare_condition(clean, prompt)

                # Sample random timestep
                while True:
                    mid_timestep = random.randint(cfg.reward.timestep_sample[0], cfg.reward.timestep_sample[1])
                    if mid_timestep in range(cfg.reward.dwt_loss_step[0], cfg.reward.dwt_loss_step[1] + 1):
                        break
                    if mid_timestep in range(cfg.reward.image_reward_loss_step[0],
                                             cfg.reward.image_reward_loss_step[1] + 1):
                        break

                # Rewritten DDIM sampling function with gradient truncation
                z, reg_loss = sampler.sample_with_mid_timestep(
                    model=cldm,
                    device=device,
                    steps=50,
                    mid_timestep=mid_timestep,
                    x_size=(len(gt), *z_0.shape[1:]),
                    cond=cond,
                    uncond=None,
                    cfg_scale=1.0,
                    cldm=cldm,
                )
                x_pred = cldm.vae_decode(z)

                # Calculate losses
                loss = 0
                loss += reg_loss
                dwt_loss_value = None
                image_reward_loss_value = None
                loss_lpips = None

                # DWT loss
                if cfg.reward.dwt_loss_step[0] <= mid_timestep <= cfg.reward.dwt_loss_step[1]:
                    dwt_loss_value = dwt_loss(((x_pred + 1) / 2).clamp(0, 1), gt / 2 + 0.5)
                    loss += dwt_loss_value

                # Image reward loss
                if cfg.reward.image_reward_loss_step[0] <= mid_timestep <= cfg.reward.image_reward_loss_step[1]:
                    x_var = torchvision.transforms.Resize(target_size)(((x_pred + 1) / 2).clamp(0, 1))
                    x_var = normalize(x_var).to(weight_dtype)
                    caption_tokens = tokenizer(caption)
                    caption_tokens = caption_tokens.to(accelerator.device)

                    outputs = reward_model(x_var, caption_tokens)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits = image_features @ text_features.T
                    scores = torch.diagonal(logits)
                    reward_loss = 1.0 - scores

                    image_reward_loss_value = reward_loss.mean() * cfg.reward.image_reward_loss_weight
                    loss += image_reward_loss_value

                    loss_lpips = net_lpips(x_pred, gt).mean() * cfg.reward.lpips_weight
                    loss += loss_lpips

                # Optimization step
                opt.zero_grad()
                accelerator.backward(loss)
                opt.step()

                # Save generated images for adversarial training
                x_pred_detached = x_pred.detach()

                # Update FRM
                if (global_step % update_freq == 0 and
                        cfg.reward.image_reward_loss_step[0] <= mid_timestep <= cfg.reward.image_reward_loss_step[1]):

                    reward_model.train()
                    for param in reward_model.parameters():
                        param.requires_grad = True
                    reward_model.lock_image_tower(unlocked_groups=20, freeze_bn_stats=False)
                    reward_model.lock_text_tower(unlocked_layers=11, freeze_layer_norm=False)

                    # Prepare data for adversarial training
                    im_pix = torch.cat((x_pred_detached, gt), dim=0)
                    N = im_pix.shape[0]
                    labels = torch.zeros(N, device=accelerator.device, dtype=weight_dtype)
                    labels[N // 2:] = 1

                    # Process images and make predictions
                    im_pix = ((im_pix.to(weight_dtype) / 2) + 0.5).clamp(0, 1)
                    x_var = torchvision.transforms.Resize(target_size)(im_pix)
                    x_var = normalize(x_var).to(im_pix.dtype)

                    caption_tokens = tokenizer(caption)
                    caption_tokens = caption_tokens.to(accelerator.device)
                    output = reward_model(x_var, caption_tokens)
                    image_features, text_features, logit_scale = (
                        output["image_features"],
                        output["text_features"],
                        output["logit_scale"]
                    )
                    logits_per_text = logit_scale * text_features @ image_features.T
                    loss_reward_op = hps_loss_ru(logits_per_text, labels) * 0.1

                    # Adversarial training optimization step
                    optimizer_reward.zero_grad()
                    accelerator.backward(loss_reward_op)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(reward_model.parameters(), 1.0)
                    optimizer_reward.step()
                    lr_scheduler_reward.step()

            # On the last step of gradient accumulation, accelerator automatically updates global_step
            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process:
                    # Log individual losses to wandb
                    log_dict = {}
                    log_dict['reg_loss'] = reg_loss.item()
                    if dwt_loss_value is not None:
                        log_dict["dwt_loss"] = dwt_loss_value.item()
                    if image_reward_loss_value is not None:
                        log_dict["image_reward_loss"] = image_reward_loss_value.item()
                    if loss_lpips is not None:
                        log_dict["lpips_loss"] = loss_lpips.item()
                    if (global_step % update_freq == 1 and 'loss_reward_op' in locals() and
                            cfg.reward.image_reward_loss_step[0] <= mid_timestep <= cfg.reward.image_reward_loss_step[
                                1]):
                        log_dict["reward_model_loss"] = loss_reward_op.item()
                    if log_dict:
                        accelerator.log(log_dict, step=global_step)

                    # Update progress bar with individual losses
                    desc_parts = [f"Epoch: {epoch:04d}", f"Global Step: {global_step:07d}"]
                    desc_parts.append(f"REG: {reg_loss.item():.6f}")
                    if dwt_loss_value is not None:
                        desc_parts.append(f"DWT: {dwt_loss_value.item():.6f}")
                    if image_reward_loss_value is not None:
                        desc_parts.append(f"FaceReward: {image_reward_loss_value.item():.6f}")
                    if loss_lpips is not None:
                        desc_parts.append(f"LPIPS: {loss_lpips.item():.6f}")
                    if (global_step % update_freq == 1 and 'loss_reward_op' in locals() and
                            cfg.reward.image_reward_loss_step[0] <= mid_timestep <= cfg.reward.image_reward_loss_step[
                                1]):
                        desc_parts.append(f"RewardOP: {loss_reward_op.item():.6f}")
                    pbar.update(1)
                    pbar.set_description(", ".join(desc_parts))

                    # Save checkpoint
                    if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                        checkpoint = pure_cldm.controlnet.state_dict()
                        ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
                        torch.save(checkpoint, ckpt_path)

            accelerator.wait_for_everyone()

            # Break if maximum steps reached
            if global_step == max_steps:
                break

        pbar.close()
        epoch += 1

    if accelerator.is_main_process:
        print("Training completed!")
        wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)