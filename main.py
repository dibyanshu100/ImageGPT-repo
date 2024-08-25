import yaml
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from utils import Kmeans, quantize, convert_to_sequence, sample_image, set_optimizer, set_scheduler
import matplotlib.pyplot as plt
import torchvision
import torch.distributed
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import IGPTModel, Config

if __name__ == '__main__':
    with open('config.yaml') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        config = Config(config)

    # model, optimizer, scheduler
    learning_rate = config.learning_rate
    model = IGPTModel(config)
    model.to(config.device)
    model_device = next(model.parameters()).device
    print(f"Model is on device: {model_device}")

    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = set_optimizer(config, model)
    scheduler = set_scheduler(config, optimizer)

    # DDP
    # model = model.to(args.local_rank)
    # model = DDP(model, device_ids=[args.local_rank])

    # dataloader, clusters
    train_dataset = torchvision.datasets.MNIST('data/', train=True, download=True,
                                               transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST('data/', train=False, download=True,
                                               transform=torchvision.transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    num_batches = len(train_loader)*10
    print(f"Number of batches: {num_batches}")


    if config.make_cluster:
        Kmeans(datasets=train_dataset, n_clusters=config.n_clusters)

    clusters = torch.from_numpy(np.load('mnist_clusters.npy')).float().to(config.device)
    # print(clusters)
    # print(clusters.shape)



    train_losses = []
    test_losses = []
    epochs = []
    for epoch in range(config.epochs):
        model.train()
        epoch_train_loss = 0
        for i, (batch, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()
            quantized_batch = quantize(batch, clusters)      # (batch_size, width, height)
            sequence = convert_to_sequence(quantized_batch)  # (batch_size, width*height)
            sequence = sequence.to(config.device)
            logits = model(sequence)                         # (batch_size, seq_len, n_clusters)
            logits = logits.view(-1, logits.size(-1))        # (batch_size*seq_len, n_clusters)
            label = sequence.view(-1)                        # (batch_size*seq_len, 1)
            loss = F.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            # tqdm.write(f"loss: {loss}")
        train_losses.append(epoch_train_loss / len(train_loader))
        epochs.append(epoch)

        # Evaluate on test set
        model.eval()
        epoch_test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                quantized_batch = quantize(batch, clusters)
                sequence = convert_to_sequence(quantized_batch)
                sequence = sequence.to(config.device)
                logits = model(sequence)
                logits = logits.view(-1, logits.size(-1))
                label = sequence.view(-1)
                loss = F.cross_entropy(logits, label)
                epoch_test_loss += loss.item()
        test_losses.append(epoch_test_loss / len(test_loader))



    # Sample an image
    sampled_image = sample_image(model, config, clusters, device = config.device)
    print(sampled_image)

    # Convert to a format suitable for visualization (e.g., [0, 1] range)
    plt.imshow(sampled_image, cmap='gray')
    plt.show()

    # loss graph
    plt.plot(epochs,train_losses)
    plt.show()



