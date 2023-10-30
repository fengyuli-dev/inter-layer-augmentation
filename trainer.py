import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from resnet import ResNet18_Cifar10
from vit import ViT


def load_cifar10(batch_size, num_workers=0):
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    cifar10_train_dataset = CIFAR10(
        root="data", train=True, download=True, transform=transform
    )
    cifar10_train_dataloader = DataLoader(
        cifar10_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    cifar10_val_dataset = CIFAR10(
        root="data", train=False, download=True, transform=transform
    )
    cifar10_val_dataloader = DataLoader(
        cifar10_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return cifar10_train_dataloader, cifar10_val_dataloader


def load_imagenet1k(batch_size, dataset_dir="/share/cuvl/datasets/imagenet"):
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    imagenet1k_train_dataset = datasets.ImageFolder(
        dataset_dir + "/train", transform=transform
    )
    imagenet1k_val_dataset = datasets.ImageFolder(
        dataset_dir + "/val", transform=transform
    )
    imagenet1k_train_dataloader = DataLoader(
        imagenet1k_train_dataset, batch_size=batch_size, shuffle=True
    )
    imagenet1k_val_dataloader = DataLoader(
        imagenet1k_val_dataset, batch_size=batch_size, shuffle=False
    )
    return imagenet1k_train_dataloader, imagenet1k_val_dataloader


def train_one_epoch(
    model,
    train_dataloader,
    loss_fn,
    optimizer,
    lr_scheduler,
    device,
    prev_validation_loss=None,
):
    model.train()
    avg_loss = 0
    for images, labels in tqdm(train_dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        lr_scheduler.step(prev_validation_loss)
    elif lr_scheduler:
        lr_scheduler.step()
    avg_loss /= len(train_dataloader)
    current_lr = optimizer.param_groups[0]["lr"]
    return avg_loss, current_lr


def validate(model, test_dataloader, loss_fn, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    avg_loss = 0
    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="Validating"):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            avg_loss += loss_fn(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total * 100, avg_loss / len(test_dataloader)


if __name__ == "__main__":
    device = torch.device("cuda:0")

    # Hyperparameters
    num_epochs = 250
    batch_size = 128
    lr = 0.1
    momentum = 0.9
    weight_decay = 1e-4

    train_dataloader, val_dataloader = load_cifar10(batch_size)
    # model = ResNet18_Cifar10().to(device)
    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
    ).to(device)
    print(f"Using {torch.cuda.device_count()} GPUs with total batch size {batch_size}")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

    best_epoch = {
        "epoch": 0,
        "val_loss": 0,
        "val_accuracy": 0,
    }
    for epoch in range(num_epochs):
        val_loss = float("inf")
        avg_loss, current_lr = train_one_epoch(
            model,
            train_dataloader,
            loss_fn,
            optimizer,
            lr_scheduler,
            device,
            prev_validation_loss=val_loss,
        )
        accuracy, val_loss = validate(model, val_dataloader, loss_fn, device)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict()
                if lr_scheduler
                else None,
            },
            f"experiments/ckpt_epoch_{epoch + 1}.pt",
        )
        if accuracy > best_epoch["val_accuracy"]:
            best_epoch["epoch"] = epoch
            best_epoch["val_loss"] = val_loss
            best_epoch["val_accuracy"] = accuracy
        print(
            f"Epoch: {epoch + 1}; lr: {current_lr:.5f}; train loss: {avg_loss:.4f}; val loss: {val_loss:.4f}; val acc: {accuracy:.2f}; best acc: {best_epoch['val_accuracy']:.2f}"
        )
    print(best_epoch)