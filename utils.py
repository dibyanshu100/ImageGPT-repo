import os
import numpy as np
import math
import torch
import logging
from sklearn.cluster import KMeans
from torchvision import datasets, transforms
import platform
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


logger = logging.getLogger(__name__)


def set_optimizer(config, model):
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay_params = set()
    no_decay_params = set()
    decay_modules = torch.nn.Linear
    no_decay_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

    for module_name, module in model.named_modules():
        for param_name, parameter in module.named_parameters():
            full_param_name = f'{module_name}.{param_name}' if module_name else param_name

            if param_name.endswith('bias'):
                no_decay_params.add(full_param_name)
            elif param_name.endswith('weight') and isinstance(module, decay_modules):
                decay_params.add(full_param_name)
            elif param_name.endswith('weight') and isinstance(module, no_decay_modules):
                no_decay_params.add(full_param_name)
    # special case for start_of_image
    no_decay_params.add('start_of_image')

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay_params & no_decay_params
    union_params = decay_params | no_decay_params
    assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, \
        f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay_params))], "weight_decay": config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay_params))], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=config.betas)
    return optimizer


def learning_rate_schedule(warmup_steps, total_steps):
    def learning_rate_fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return learning_rate_fn

def set_scheduler(config, optimizer):
    scheduler = {
        "scheduler": LambdaLR(
            optimizer, learning_rate_schedule(config.warmup_steps, config.total_steps)
        ),
        "interval": "step",
    }
    return scheduler



def Kmeans(datasets, n_clusters=10):
    #Load dataset
    images = datasets.data.numpy()                        # (60000, 28, 28)
    flattened_images = images.reshape(-1, 1)  # (60000*28*28,)
    flattened_images = flattened_images / 255.0             # Normalising
    n_clusters = n_clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(flattened_images)
    clusters = kmeans.cluster_centers_
    np.save('mnist_clusters.npy', clusters)  # Saving the cluster centroids as a .npy file
    print(f'Cluster centroids saved to mnist_clusters.npy')


def squared_euclidean_distance(images, clusters):
    """This function calculates the squared Euclidean distance between each
    image pixel (flattened into a vector) and the cluster centroids."""

    # (channels, n_clusters) -> (n_clusters, channels)
    clusters = torch.transpose(clusters, 0, 1)
    squared_logits = torch.sum(torch.square(images), dim=1, keepdim=True)
    squared_clusters = torch.sum(torch.square(clusters), dim=0, keepdim=True)
    mul_logits_clusters = torch.matmul(images, clusters)
    euclidean_distance = squared_logits - 2 * mul_logits_clusters + squared_clusters
    return euclidean_distance


def quantize(images, clusters):
    """
    This function quantize the image by assigning each pixel to the nearest cluster,
    based on the squared Euclidean distance computed by the previous function.
    """
    images = images.to(clusters.device)
    clusters = clusters.to(images.device)
    batch_size, channels, height, width = images.shape
    images = images.permute(0, 2, 3, 1).contiguous()
    images = images.view(-1, channels)  # flatten to pixels
    distance = squared_euclidean_distance(images, clusters)
    quantized = torch.argmin(distance, 1)
    quantized = quantized.view(batch_size, height, width)
    return quantized


def convert_to_sequence(images):
    pixels = images.view(images.shape[0], -1)  # flatten images into sequences
    return pixels


def unquantize(x, centroids):
    if isinstance(centroids, np.ndarray):
        centroids = torch.tensor(centroids, dtype=torch.float32, device=x.device)

    return centroids[x]


def sample_image(model, config, clusters, device='mps'):

    model.eval()
    # Initialize the sampled image tensor with zeros
    sampled_image = torch.zeros((1, config.n_pixels, config.n_pixels), dtype=torch.float, device=device)
    # Shape: (1, 28, 28)

    with torch.no_grad():
        for i in range(config.n_pixels * config.n_pixels):
            # Quantize
            quantized_image = quantize(sampled_image.unsqueeze(0), clusters).to(device)
            # Flatten the sampled image to a sequence
            sequence = quantized_image.view(1, -1)                                       # (1, 784)
            print("Seq: ",sequence)
            # Generate logits from the model
            logits = model(sequence)                                                     # (1, 784, n_clusters)
            print("Logits: ",logits)
            # Extract logits for the current pixel position
            logits_for_current_pixel = logits[:, i, :]                                   # (1, n_clusters)
            # Apply temperature scaling to logits
            temperature = 1.0
            probs = torch.softmax(logits_for_current_pixel / temperature, dim=-1)        # (1, n_clusters)
            # Sample the next pixel from the distribution
            cluster_index = torch.multinomial(probs, num_samples=1).squeeze().item()     # Sampled cluster index (0,)
            # print("cluster index: ",cluster_index)
            # Unquantize the pixel to get the original value
            pixel_value = unquantize(cluster_index, clusters)                            # (H, W)
            pixel_value = torch.tensor(pixel_value, dtype=torch.float, device=device)
            # Update the sampled image with the new pixel value
            row, col = divmod(i, config.n_pixels)                                        # (row, col)
            sampled_image[0, row, col] = pixel_value.item()                              # (1, 28, 28)


    return sampled_image.squeeze().cpu().numpy()
