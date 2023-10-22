import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import argparse
import os
import shutil

try:
    import lpips
except:
    pass

from torchvision.utils import make_grid

from renderer.sagan_nr import Discriminator, Generator
from renderer.renderer import EmojiRenderer

from accelerate.utils import ProjectConfiguration
from accelerate import Accelerator


class MaskedLoss(nn.Module):
    def __init__(
        self,
        mask_multiplier,
        device,
        max_epochs,
        img_size=64,
        criterion=F.mse_loss,
        use_perceptual=False,
    ):
        super().__init__()
        self.mask_multiplier = mask_multiplier
        self.device = device
        self.max_epochs = max_epochs
        self.img_size = img_size
        self.use_perceptual = use_perceptual
        self.criterion = criterion
        if use_perceptual:
            self.perceptual = lpips.LPIPS(net="alex", verbose=False)

    def forward(self, input, target, epoch):
        mask = torch.mean(target, dim=1)

        masked_weight = (-9 * epoch / self.max_epochs) + self.mask_multiplier
        whitespace = torch.eq(mask, torch.ones_like(mask))
        mask = torch.where(
            whitespace,
            torch.zeros_like(mask, device=self.device),
            torch.ones_like(mask, device=self.device) * masked_weight,
        )
        mask = mask.view(-1, 1, self.img_size, self.img_size)
        loss = torch.mean(self.criterion(input, target, reduction="none") * mask)
        if self.use_perceptual:
            loss += torch.mean(self.perceptual(input, target) * mask)

        return loss


def train(config, args):
    accelerate_config = ProjectConfiguration(
        total_limit=config["total_limit"],
    )
    accelerator = Accelerator(
        cpu=args.cpu,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=args.logging_dir,
        project_config=accelerate_config,
    )

    generator = Generator(image_size=config["img_size"], z_dim=128)
    discriminator = Discriminator(image_size=config["img_size"])
    er = EmojiRenderer(canvas_size=config["img_size"])

    criterion_reconstruction = MaskedLoss(
        config["mask_weight"],
        accelerator.device,
        config["num_epochs"],
        img_size=config["img_size"],
    )

    optimizer_generator = AdamW(
        generator.parameters(),
        lr=config["lr_generator"],
        betas=(config["adam_beta"], 0.999),
    )

    optimizer_discriminator = AdamW(
        discriminator.parameters(),
        lr=config["lr_discriminator"],
        betas=(config["adam_beta"], 0.999),
    )

    (
        generator,
        discriminator,
        optimizer_generator,
        optimizer_discriminator,
        criterion_reconstruction,
    ) = accelerator.prepare(
        generator,
        discriminator,
        optimizer_generator,
        optimizer_discriminator,
        criterion_reconstruction,
    )

    if not args.no_tracking:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run, config)

    start_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [os.path.join(args.output_dir, f.name) for f in os.scandir(args.output_dir) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]
        training_difference = os.path.splitext(path)[0]

        if training_difference.startswith("epoch_"):
            start_epoch = int(training_difference.replace("epoch_", "")) + 1

        elif training_difference.startswith("checkpoint_"):
            start_epoch = (
                int(training_difference.replace("checkpoint_", "")) + 1
            ) * config["checkpoint_every_n_steps"] + 1

        accelerator.print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, config["num_epochs"]):
        optimizer_generator.zero_grad(set_to_none=True)
        optimizer_discriminator.zero_grad(set_to_none=True)

        emoji_idx = torch.randint(
            0, 4099, (config["batch_size"],), device=accelerator.device
        )
        pos = torch.empty(
            (config["batch_size"], 4), device=accelerator.device
        ).uniform_(-1, 1)

        # Calculate discriminator loss on actual
        if epoch % 2 == 0:
            discriminator.train()
            generator.eval()

            actual = er.generate_batch(list(zip(emoji_idx, pos)), accelerator.device)

            discriminator_actual, *_ = discriminator(actual, emoji_idx, pos)
            loss_discriminator_actual = F.relu(1.0 - discriminator_actual).mean()

            with torch.no_grad():
                rendered, *_ = generator(emoji_idx, pos)

            discriminator_rendered, *_ = discriminator(
                rendered.detach(), emoji_idx, pos
            )
            loss_discriminator_rendered = F.relu(1.0 + discriminator_rendered).mean()

            accelerator.backward(
                loss_discriminator_actual + loss_discriminator_rendered
            )
            optimizer_discriminator.step()
        # Calculate generator loss
        else:
            discriminator.eval()
            generator.train()

            rendered, *_ = generator(emoji_idx, pos)

            loss_generator_reconstruction = torch.tensor([0])

            loss_generator_discriminator = -torch.mean(
                discriminator(rendered, emoji_idx, pos)[0]
            )

            accelerator.backward(
                loss_generator_discriminator
            )
            optimizer_generator.step()

        if epoch % config["checkpoint_every_n_steps"] == 0 and epoch != 0:
            # if epoch < 200000:
            #     lr = 1e-4
            # elif epoch < 400000:
            #     lr = 1e-5
            # else:
            #     lr = 1e-6
            # for param_group in p[optimizer_nr].param_groups:
            #     param_group["lr"] = lr

            accelerator.get_tracker("tensorboard").tracker.add_image(
                "rendered", make_grid(F.sigmoid(rendered[:4]), normalize=True), epoch
            )
            accelerator.get_tracker("tensorboard").tracker.add_image(
                "actual", make_grid(actual[:4], normalize=True), epoch
            )

            accelerator.print(
                f"epoch {epoch}, loss generator reconstruction: {loss_generator_reconstruction.item()}, loss generator discriminator: {loss_generator_discriminator.item()}, loss discriminator actual: {loss_discriminator_actual.item()}, loss discriminator rendered: {loss_discriminator_rendered.item()}"
            )
            if not args.no_tracking:
                accelerator.log(
                    {
                        "loss generator reconstruction": loss_generator_reconstruction.item(),
                        "loss generator discriminator": loss_generator_discriminator.item(),
                        "loss discriminator actual": loss_discriminator_actual.item(),
                        "loss discriminator rendered": loss_discriminator_rendered.item(),
                        "epoch": epoch,
                    },
                    step=epoch,
                )

            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

            if len(os.listdir(args.output_dir)) > config["total_limit"]:
                dirs = [os.path.join(args.output_dir, f.name) for f in os.scandir(args.output_dir) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                for i in range(len(dirs) - config["total_limit"]):
                    shutil.rmtree(dirs[i])

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
        "num_epochs": int(6e5),
        "img_size": 64,
        "mask_weight": 10,
        "adam_beta": 0.5,
        "lr_generator": 1e-4,
        "lr_discriminator": 4e-4,
        "batch_size": 64,
        "checkpoint_every_n_steps": 500,
        "total_limit": 5
    }

    train(config, args)
