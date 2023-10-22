from PIL import Image
import os

import torch
from torchvision.transforms import ToTensor, Normalize, Compose
from multiprocessing import Pool, cpu_count


class EmojiRenderer:
    def __init__(
        self, canvas_size: int = 256, generate_single=True, generate_alpha=False
    ):
        self.canvas_size = canvas_size
        self.generate_alpha = generate_alpha
        self.clear()
        self.emojis_dir = os.path.abspath("./emojis")
        self.all_emojis = os.listdir(self.emojis_dir)
        self.all_emojis.sort()

        self.total_emojis = len(self.all_emojis)

        self.min_angle = -45
        self.max_angle = 45

        self.min_resize = 0.9
        self.max_resize = 1.1

        self.transform = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        self.generate_single = generate_single

        self.bbox = None

    def unnormalize(self, value, min_target, max_target):
        return (value / 2 + 0.5) * (max_target - min_target) + min_target

    def normalize(self, value, min_target, max_target):
        return ((value - min_target) / (max_target - min_target) - 0.5) * 2

    def draw(self, emoji_idx, resize, location_x, location_y, rotation=0):
        """Draws to the existing canvas. Expects the provided data to be in the range [-1,1]"""
        if self.generate_single:
            self.clear()

        emoji = self.all_emojis[int(emoji_idx)]
        rotation = int(self.unnormalize(rotation, self.min_angle, self.max_angle))
        resize = self.unnormalize(resize, self.min_resize, self.max_resize)

        emoji_img = Image.open(os.path.join(self.emojis_dir, emoji)).convert("RGBA")
        emoji_img = emoji_img.rotate(rotation, expand=1)

        width, height = emoji_img.size
        width, height = int((self.canvas_size // 4) * resize), int(
            (self.canvas_size // 4) * resize
        )
        emoji_img = emoji_img.resize((width, height))

        paste_x = int(
            self.unnormalize(location_x, -width // 2, self.canvas_size - (width // 2))
        )
        paste_y = int(
            self.unnormalize(location_y, -height // 2, self.canvas_size - (height // 2))
        )

        self.canvas.paste(emoji_img, (paste_x, paste_y), emoji_img)

        # self.bbox = (max(paste_y, 0), min(paste_y + width, self.canvas_size),
        #                 max(paste_x, 0), min(paste_x + height, self.canvas_size))

    def clear(self):
        if self.generate_alpha:
            self.canvas = Image.new(
                "RGBA", (self.canvas_size, self.canvas_size), (255, 255, 255, 255)
            )
        else:
            self.canvas = Image.new(
                "RGB", (self.canvas_size, self.canvas_size), (255, 255, 255)
            )

    def get_canvas(self):
        return self.canvas

    def get_torch_canvas(self):
        return self.transform(self.canvas).unsqueeze(0)

    def generate_batch(self, inputs, device: torch.device):
        batch = []

        for i in range(len(inputs)):
            if len(inputs[i]) == 2:
                emoji_idx, params = inputs[i]
                emoji_idx = emoji_idx.item()
                resize, location_x, location_y, rotation = [p.item() for p in params]
            else:
                emoji_idx, resize, location_x, location_y, rotation = [
                    i.item() for item in inputs[i]
                ]

            self.draw(emoji_idx, resize, location_x, location_y, rotation)
            batch.append(self.get_torch_canvas())
        return torch.cat(batch, dim=0).to(device)
