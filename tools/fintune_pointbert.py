import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import argparse
from tqdm import tqdm
from load_pointbert import load_pointbert, pointbert_cls
from load_modelnet40 import ModelNetDataset

def load_dataset(data_path:str, batch_size:int):
    TRAIN_DATASET = ModelNetDataset(
        root=data_path,
        npoint=8192,
        split='train',
        normal_channel=False
    )
    
    train_dataLoader = DataLoader(
        TRAIN_DATASET,
        batch_size=batch_size,
        shuffle=True,
        num_workers=64
    )
    
    TEST_DATASET = ModelNetDataset(
        root=data_path,
        npoint=8192,
        split='test',
        normal_channel=False
    )
    
    test_dataLoader = DataLoader(
        TEST_DATASET,
        batch_size=batch_size,
        shuffle=False,
        num_workers=64
    )
    
    return TRAIN_DATASET, TEST_DATASET, train_dataLoader, test_dataLoader

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating"):
            points, labels = data
            points, labels = points.cuda(), labels.cuda().squeeze().long()
            points = F.pad(points, (0, 3))

            # forward
            outputs = model(points)
            preds = outputs.argmax(dim=1)

            # update accuracy
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Evaluation Accuracy: {acc * 100:.2f}%")
    return acc

def show_model_parameters_info(model:nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percent = 100 * trainable_params / total_params if total_params > 0 else 0

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {percent:.2f}%")

def main(batch_size:int, epoch:int, learning_rate:float, config_path: str, ckpt_path: str, data_path: str):
    train_dataset, test_dataset, train_dataLoader, test_dataLoader = load_dataset(data_path, batch_size)
    
    pointbert = load_pointbert(config_path,
                               ckpt_path, use_color=True)

    pointbert = pointbert_cls(pointbert, train_dataset.classes.__len__()).cuda()
    show_model_parameters_info(pointbert)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pointbert.parameters(), lr=learning_rate)

    # 训练循环
    pointbert.train()
    for epoch in range(epoch):  # 你可以设置更长
        total_loss = 0
        correct = 0
        total = 0
        for points, label in tqdm(train_dataLoader, desc=f"Epoch {epoch+1}"):
            points = points.cuda()
            points = F.pad(points, (0, 3))
            
            label = label.cuda().squeeze().long()

            optimizer.zero_grad()
            output = pointbert(points)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = output.argmax(dim=-1)
            correct += (preds == label).sum().item()
            total += label.size(0)

        acc = 100 * correct / total
        print(f"[Epoch {epoch+1}] Total loss: {total:.4f}, Accuracy: {acc:.2f}%")
        torch.save(pointbert.state_dict(), "pointbert_finetuned.pth")
        evaluate(pointbert, test_dataLoader)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training or evaluation.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Size of a single batch")
    parser.add_argument("--epoch", type=int, default=40,
                        help="Epoch")
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--config-path", type=str, default="pointllm/model/pointbert/PointTransformer_8192point_2layer.yaml",
                        help="Path to config file")
    parser.add_argument("--ckpt-path", type=str, default="RunsenXu/PointLLM_13B_v1.2/pytorch_model-00006-of-00006.bin",
                        help="Path to model weights")
    parser.add_argument("--data-path", type=str, default="data/modelnet40_normal_resampled",
                        help="Path to dataset")

    args = parser.parse_args()
    main(args.batch_size, args.epoch, args.learning_rate, args.config_path, args.ckpt_path, args.data_path)
