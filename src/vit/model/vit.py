import torch.nn as nn
import torch
from .block import TransformerBlockStack
from .patcher import Patcher
import torchvision

class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.resnet = self.create_resnet(config.resnet_layers)
        self.resnet = self.resnet.to(config.device)
        self.positional_embedding_layer = nn.Embedding(config.num_patches + 1, config.hidden_dim)
        self.positional_embedding_layer.weight.data.uniform_(0, 1)

        self.encoder = TransformerBlockStack(config)
        self.dense = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.device = config.device
        self.patch_embedding_linear = nn.Linear(config.final_resnet_output_dim, config.hidden_dim, bias=False).to(config.device)

    def create_resnet(self, n_layer):
        model = torchvision.models.resnet152(weights='ResNet152_Weights.IMAGENET1K_V1')
        return torch.nn.Sequential(*(list(model.children())[:-n_layer]))

    def create_patches(self, in_channels):
        batch, channel, height, width = in_channels.shape
        in_channels = in_channels.view(batch, height * width, channel)

        in_channels = in_channels.to(self.device)
        patch_embeddings = self.patch_embedding_linear(in_channels)

        class_token = nn.Parameter(torch.zeros(batch, 1, patch_embeddings.shape[-1]))  # Learnable class token
        return torch.cat((class_token.to(self.device), patch_embeddings), dim=1)

    def build_positional_embedding(self, patch_embeddings):
        batch, patches, embed_dim = patch_embeddings.shape
        positions = torch.arange(0, patches).view(1, -1).repeat(batch, 1)
        return self.positional_embedding_layer(positions.to(self.device))

    def forward(self, x):
        features = self.resnet(x)
        patch_embeddings = self.create_patches(features)
        positional_embeddings = self.build_positional_embedding(patch_embeddings)
        patch_embeddings = patch_embeddings + positional_embeddings
        encoded_output, attention_weights_list = self.encoder(patch_embeddings)
        encoded_features = self.dense(encoded_output.mean(dim=1))
        return encoded_features, attention_weights_list
