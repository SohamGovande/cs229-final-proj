import torch.nn as nn

class Patcher(nn.Module):
    def __init__(self, patch_size):
        super(Patcher, self).__init__()
        self.patch_size = patch_size

    def forward(self, images):
        if images.dim() == 3:
            images = images.unsqueeze(0)

        batch_size, channels, height, width = images.size()
        patch_height, patch_width = self.patch_size

        num_patches_height = height // patch_height
        num_patches_width = width // patch_width
        num_patches = num_patches_height * num_patches_width

        patches = images.unfold(2, patch_height, patch_height).unfold(3, patch_width, patch_width)
        patches = patches.contiguous().view(batch_size, channels, -1, patch_height, patch_width)
        patches = patches.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, num_patches, -1)

        return patches
