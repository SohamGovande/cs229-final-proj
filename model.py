import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import nn
import torchvision.models as models


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_dim, config.mlp_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.out = nn.Linear(config.mlp_dim, config.hidden_dim)

    def forward(self, x):
        x = F.gelu(self.dense(x))
        return self.out(self.dropout(x))


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(config.hidden_dim, config.num_heads)
        self.mlp = MLP(config)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.attention_weights = None

    def forward(self, x):
        n_x = self.norm1(x)
        attn_output, self.attention_weights = self.attention(n_x, n_x, n_x)
        x = x + attn_output
        return x + self.mlp(self.norm2(x))


class EncoderStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([Encoder(config) for _ in range(config.num_layers)])

    def forward(self, x):
        attention_weights = []
        for layer in self.layers:
            x = layer(x)
            attention_weights.append(layer.attention_weights)
        return x, attention_weights


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


class VisionTransformerResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.resnet = self.create_resnet(config.resnet_layers)
        self.resnet = self.resnet.to(config.device)
        self.positional_embedding_layer = nn.Embedding(config.num_patches + 1, config.hidden_dim)
        self.positional_embedding_layer.weight.data.uniform_(0, 1)

        self.encoder = EncoderStack(config)
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


# Vision Transformer for models without ResNet
class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.positional_embedding_layer = nn.Embedding(config.num_patches, config.hidden_dim).to(config.device)
        self.positional_embedding_layer.weight.data.uniform_(0, 1)

        self.encoder = EncoderStack(config)
        self.dense = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.device = config.device
        self.patch_linear_proj = nn.Linear(config.patching_elements, config.hidden_dim, bias=False).to(config.device)
        self.patcher = Patcher((config.patch_size, config.patch_size))
        self.positional_embeddings = self.build_positional_embedding(config.num_patches, config.hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def create_patches(self, x):
        batch = x.shape[0]
        patches = self.patcher(x).to(self.device)
        patches = self.patch_linear_proj(patches)
        return patches

    def build_positional_embedding(self, num_patches, embed_dim):
        positions = torch.arange(0, num_patches).view(1, -1).to(self.device)
        return self.positional_embedding_layer(positions)
        
    def forward(self, x):
        batch_size = x.shape[0]
        patch_embeddings = self.create_patches(x)
        positional_embeddings = self.positional_embeddings.repeat(batch_size, 1, 1)
        patch_embeddings = patch_embeddings + positional_embeddings
        encoded_output, attention_weights_list = self.encoder(patch_embeddings)
        encoded_output = encoded_output.mean(dim=1)
        encoded_output = self.dense(encoded_output)
        return encoded_output, attention_weights_list

class DenseNet121Baseline(nn.Module):
    def __init__(self, num_classes=2):
        super(DenseNet121Baseline, self).__init__()
        self.model = models.densenet121(pretrained=True)
        # Replace the classifier with a new one (since we have 2 classes)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)