# Import necessary dependencies
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from const import *
import matplotlib.pyplot as plt

from patch_embedding import PatchEmbedding
from encoder import Encoder
from vit import VIT

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Helper function for `train_model()`.
    """
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """
    Helper function for `train_model()`.
    """
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def train_model(
    model: VIT,
    optimizer: torch.optim.Adam,
    criterion: torch.nn.CrossEntropyLoss,
    train_loader: DataLoader,
    test_loader: DataLoader,
    training_loop_counter: int = 0,
    num_epochs: int = NUM_EPOCHS,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    device: torch.cuda.device = torch.device('cpu')
):
    """
    Complete one training iteration of the given model. At the end the checkpoint of best model will be saved as well
    as image showing the graphs of loss and accuracy progression on train and test sets.
    
    Parameters
    ----------
    `model`: VIT
        Vision Transformer model with defined hyperparameters.
    `optimizer`: torch.nn.Adam
        Optimizer (In our case - `torch.nn.Adam`) with defined hyperparameters.
    `train_loader`: DataLoader
        Train data loader.
    `test_loader`: DataLoader
        Test data loader.
    `num_epochs`: int = 30
        The amount of epochs in one training iteration.
    `early_stopping_patience`: int = 10
        The amount of epoch with no loss improvement. If such amount is reached, training is stopped.
    `device`: torch.cuda.device = torch.cuda.device('cpu')
        Device which will be used for training, can either be `torch.cuda.device('cpu')` or `torch.cuda.device('cuda:0')`.
    
    Returns
    ----------
    `None`
    """
    
    for images, labels in train_loader:
        img_size = images.shape[-1]
        break
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, test_loader, criterion, device)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)


        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            }, f'best_model_{training_loop_counter}.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch+1}')
                training_loop_counter += 1
                break

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss:
              img_size={img_size}, batch_size={model.batch_size}, num_layers={model.num_layers}, num_heads={model.num_heads},
              latent_size={model.latent_size}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Acc')
    plt.plot(val_acc_history, label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(f'best_model_{training_loop_counter}.png')

    plt.show()