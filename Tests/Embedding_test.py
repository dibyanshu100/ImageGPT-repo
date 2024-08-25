# Testing Embeddings

import torch
import torch.nn as nn

# Define the configuration class
class Config:
    def __init__(self):
        self.n_clusters = 9     # Number of clusters (quantization levels)
        self.hidden_size = 8      # Dimensionality of embeddings
        self.n_pixels = 28         # For a 28x28 image (e.g., MNIST)
        self.embed_pdrop = 0.1     # Dropout probability

# Define the Embeddings class
class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        # Embedding layers
        self.pix_embed = nn.Embedding(config.n_clusters, config.hidden_size)
        print(f"Pix embed shape: {self.pix_embed}")
        self.pos_embed = nn.Embedding(config.n_pixels ** 2, config.hidden_size)
        print(f"Pos embed shape: {self.pos_embed}")
        self.embed_drop = nn.Dropout(config.embed_pdrop)

    def forward(self, x):
        print(f"x shape: {x.shape}")
        seq_length = x.size(1)
        print(f"seq_length: {seq_length}")
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        print(f"position_id: {position_ids.shape}")
        position_ids = position_ids.unsqueeze(0).expand_as(x)
        print(position_ids.shape)

        pix_embeddings = self.pix_embed(x)
        print(f"pix_embeddings shape: {pix_embeddings.shape}")
        pos_embeddings = self.pos_embed(position_ids)
        print(f"pos_embeddings shape: {pos_embeddings.shape}")
        embeddings = self.embed_drop(pix_embeddings + pos_embeddings)
        return embeddings

# Instantiate the config and Embeddings class
config = Config()
embedding_layer = Embeddings(config)

# Create a batch of quantized images
batch_size = 10  # Number of images in the batch
sequence_length = config.n_pixels ** 2  # 28x28 = 784 pixels per image

# Randomly generate quantized pixel values (cluster indices) for the batch
quantized_images = torch.randint(0, config.n_clusters, (batch_size, sequence_length))
print(quantized_images)
print(quantized_images.shape)

# Run the forward pass with the sample input
output_embeddings = embedding_layer(quantized_images)

# Output the result
print("Sample input (quantized images):", quantized_images)
print("Output embeddings shape:", output_embeddings.shape)
