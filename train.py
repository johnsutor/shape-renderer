import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import argparse
import os

try:
    import lpips
except:
    pass

from torchvision.utils import make_grid
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from renderer.modules import VQVAE
from renderer.renderer import EmojiRenderer

from accelerate import Accelerator


class MaskedLoss(nn.Module):
    def __init__(
        self,
        mask_multiplier,
        device,
        img_size=256,
        criterion=F.mse_loss,
        use_perceptual=True,
    ):
        super().__init__()
        self.mask_multiplier = mask_multiplier
        self.device = device
        self.img_size = img_size
        self.use_perceptual = use_perceptual
        self.criterion = criterion
        if use_perceptual:
            self.perceptual = lpips.LPIPS(net="alex", verbose=False)
            self.perceptual.eval()

    def forward(self, input, target):
        mask = torch.mean(target, dim=1)

        whitespace = torch.eq(mask, torch.ones_like(mask))
        mask = torch.where(
            whitespace,
            torch.ones_like(mask, device=self.device),
            torch.ones_like(mask, device=self.device) * self.mask_multiplier,
        )
        mask = mask.view(-1, 1, self.img_size, self.img_size)
        loss = torch.mean(self.criterion(input, target, reduction="none") * mask)
        if self.use_perceptual:
            loss += torch.mean(self.perceptual(input, target) * mask)

        return loss

def train(config, args):
    accelerator = Accelerator(
        cpu=args.cpu,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=args.logging_dir,
    )

    torch.backends.cudnn.benchmark = True

    vae = VQVAE()
    er = EmojiRenderer()

    criterion_reconstruction = MaskedLoss(config["mask_weight"], accelerator.device)

    optimizer_vae = AdamW(
        vae.parameters(), lr=config["lr"], betas=(config["adam_beta"], 0.999)
    )


    scheduler_vae = CosineAnnealingWarmRestarts(
        optimizer_vae, T_0=config["lr_decay_step"], T_mult=1, eta_min=1e-6
    )

    

    # TODO: Implement EMA

    (
        vae,
        optimizer_vae,
        scheduler_vae,
        criterion_reconstruction,
    ) = accelerator.prepare(
        vae,
        optimizer_vae,
        scheduler_vae,
        criterion_reconstruction,
    )

    if not args.no_tracking:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run, config)

    start_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        start_epoch = int(training_difference.replace("epoch_", "")) + 1

    for epoch in range(start_epoch, config["num_epochs"]):
        vae.train()

        optimizer_vae.zero_grad()

        emoji_idx = torch.randint(
            0, 4099, (config["batch_size"],), device=accelerator.device
        )
        pos = torch.empty(
            (config["batch_size"], 4), device=accelerator.device
        ).uniform_(-1, 1)

        actual = er.generate_batch(list(zip(emoji_idx, pos)), accelerator.device)

        rendered, _, loss_codebook = vae(actual)

        # loss_render = criterion(rendered, actual)
        loss_reconstruction = criterion_reconstruction(rendered, actual)

        # accelerator.backward(loss_render, retain_graph=True)
        if (epoch + 1) % config["gradient_update_every_n_steps"] == 0:
            accelerator.backward(
                loss_reconstruction + loss_codebook,
                retain_graph=True,
            )
            optimizer_vae.step()
            scheduler_vae.step()

        if epoch % config["checkpoint_every_n_steps"] == 0:
            accelerator.get_tracker("tensorboard").tracker.add_image(
                "rendered", make_grid(F.sigmoid(rendered[:4]), normalize=True), epoch
            )
            accelerator.get_tracker("tensorboard").tracker.add_image(
                "actual", make_grid(actual[:4], normalize=True), epoch
            )

            accelerator.print(
                f"epoch {epoch}, loss reconstruction: {loss_reconstruction.item()}, Loss codebook: {loss_codebook.item()}"
            )
            if not args.no_tracking:
                accelerator.log(
                    {   
                        "loss_codebook": loss_codebook.item(),
                        "epoch": epoch,
                        "loss_reconstruction": loss_reconstruction.item(),
                    },
                    step=epoch,
                )

            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    accelerator.print("Finished Training")
    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training arguments.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.abspath("./data"),
        help="The data folder on disk.",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="If passed, will use FP16 training."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="If passed, will train on the CPU."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--no_tracking",
        action="store_true",
        help="Whether to load in all tensorboard experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Location on where to store experiment tracking logs`",
    )
    args = parser.parse_args()

    config = {
        "num_epochs": int(4e6),
        "img_size": 256,
        "mask_weight": 10,
        "adam_beta": 0.9,
        "lr": 1e-3,
        "batch_size": 16,
        "lr_decay_step": int(5e4),
        "gradient_update_every_n_steps": 4,
        "checkpoint_every_n_steps": 1000,
    }

    train(config, args)
