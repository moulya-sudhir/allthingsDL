import os
import torch
from torch import nn, optim
import data_setup, engine, model_builder, utils
from torchvision import transforms

# Setup hyperparameters
num_epochs = 5
batch_size = 32
hidden_units = 10
lr = 0.001

# Setup directories
train_dir = '../data/pizza_steak_sushi/train'
test_dir = '../data/pizza_steak_sushi/test'

# Setup target device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

# Create dataloaders
train_loader, test_loader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=batch_size
)

# Create model
model = model_builder.TinyVGG(
    input_dim=3,
    hidden_dim=hidden_units,
    output_dim=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(), lr=lr)

# Start training
model_results = engine.train(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=device,
    epochs=num_epochs
)

# Save model
utils.save_model(
    model=model,
    target_dir='../models',
    model_name='05_going_modular_tinyvgg_model.pth'
)