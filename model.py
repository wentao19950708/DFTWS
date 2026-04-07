import os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights
import pandas as pd


class Config:

    cla_img_dir = r"D:\DeepLearning\ViT\dataset\cla\images"
    cla_label_dir = r"D:\DeepLearning\ViT\dataset\cla\labels"


    reg_img_dir = r"D:\DeepLearning\ViT\dataset\reg\images"
    reg_label_dir = r"D:\DeepLearning\ViT\dataset\reg\labels"


    cla_batch_size = 48
    reg_batch_size = 12
    epochs = 200
    learning_rate = 1e-3
    weight_decay = 5e-4

    train_ratio = 0.7


    img_size = 224
    num_classes = 4
    freeze_backbone = True
    freeze_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = "results"
    model_save_path = os.path.join(save_dir, "vit_multi_task_model.pth")
    plot_save_path = os.path.join(save_dir, "training_plot.png")
    metric_csv_path = os.path.join(save_dir, "training_metrics.csv")  # 新增：表格保存路径


    train_mode ='both'  # 'both', 'classification_only', 'regression_only'


os.makedirs(Config.save_dir, exist_ok=True)


class DualFactorLossBalancer:
    def __init__(self, num_tasks, ema_alpha=0.9, temp=0.5, abs_weight=0.8,
                 weight_smooth=0.7, max_loss_ratio=100, min_weight=0.01, epsilon=1e-8,
                 convergence_threshold=0.05, force_reduce_weight=0.1,
                 val_history_window=5,
                 min_trend_samples=20,
                 trend_smooth_factor=0.2,
                 overfit_penalty=0.5):
        self.num_tasks = num_tasks
        self.ema_alpha = ema_alpha
        self.temp = temp
        self.abs_weight = abs_weight
        self.weight_smooth = weight_smooth
        self.max_loss_ratio = max_loss_ratio
        self.min_weight = min_weight
        self.epsilon = epsilon
        self.convergence_threshold = convergence_threshold
        self.force_reduce_weight = force_reduce_weight

        self.val_history_window = val_history_window
        self.min_trend_samples = min_trend_samples
        self.trend_smooth_factor = trend_smooth_factor
        self.overfit_penalty = overfit_penalty

        self.converged_tasks = torch.zeros(num_tasks)
        self.loss_ema = torch.zeros(num_tasks)
        self.task_weights = torch.ones(num_tasks) / num_tasks


        self.val_loss_history = []
        self.loss_trends = torch.zeros(num_tasks)
        self._device = None

    def update_validation_losses(self, val_cls_loss, val_reg_loss, device):
        with torch.no_grad():
            val_losses = torch.tensor([val_cls_loss, val_reg_loss], device=device)
            if self._device is None:
                self._device = device
                self.loss_trends = self.loss_trends.to(device)


            self.val_loss_history.append(val_losses.clone())
            if len(self.val_loss_history) > self.val_history_window:
                self.val_loss_history.pop(0)


            self._update_loss_trends()
    def _update_loss_trends(self):

        if len(self.val_loss_history) < self.min_trend_samples:
            return


        num_samples = len(self.val_loss_history)
        x = torch.arange(num_samples, device=self._device).float()
        x_mean = torch.mean(x)


        for task_idx in range(self.num_tasks):

            y = torch.tensor(
                [epoch_loss[task_idx] for epoch_loss in self.val_loss_history],
                device=self._device
            ).float()
            y_mean = torch.mean(y)

            covariance = torch.sum((x - x_mean) * (y - y_mean))
            variance = torch.sum((x - x_mean) ** 2) + self.epsilon
            slope = covariance / variance


            self.loss_trends[task_idx] = (self.trend_smooth_factor * slope
                                          + (1 - self.trend_smooth_factor) * self.loss_trends[task_idx])

    def _detect_overfitting(self):

        if len(self.val_loss_history) < self.min_trend_samples:
            return torch.zeros(self.num_tasks, device=self._device)

        overfitting = torch.zeros(self.num_tasks, device=self._device)
        overfitting[self.loss_trends > 1e-6] = 1.0
        return overfitting

    def update(self, current_losses, device):
        with torch.no_grad():
            current_losses = torch.tensor(current_losses, device=device)

            if self._device is None:
                self._device = device
                self.loss_ema = self.loss_ema.to(device)
                self.task_weights = self.task_weights.to(device)
                self.converged_tasks = self.converged_tasks.to(device)

            if torch.all(self.loss_ema == 0):
                self.loss_ema = current_losses.clone()
            else:
                self.loss_ema = self.ema_alpha * self.loss_ema + (1 - self.ema_alpha) * current_losses

            convergence_rate = -(self.loss_ema - current_losses) / (self.loss_ema + self.epsilon)
            rate_factor = -convergence_rate
            min_loss = torch.min(current_losses) + self.epsilon
            loss_ratio = current_losses / min_loss
            loss_ratio = torch.clamp(loss_ratio, max=self.max_loss_ratio)
            abs_factor = 1.0 / loss_ratio

            combined_score = (1 - self.abs_weight) * rate_factor + self.abs_weight * abs_factor
            exp_values = torch.exp(combined_score / self.temp)
            new_weights = exp_values / torch.sum(exp_values)
            new_weights = self.weight_smooth * self.task_weights + (1 - self.weight_smooth) * new_weights
            new_weights = torch.clamp(new_weights, min=self.min_weight)


            self.converged_tasks = torch.zeros_like(self.converged_tasks)
            for i in range(self.num_tasks):
                if current_losses[i] < self.convergence_threshold:
                    self.converged_tasks[i] = 1.0
                    new_weights[i] *= (1 - self.force_reduce_weight)

            if len(self.val_loss_history) >= self.min_trend_samples:
                overfitting_tasks = self._detect_overfitting()

                new_weights = new_weights * (1 - overfitting_tasks * self.overfit_penalty)
                new_weights = torch.clamp(new_weights, min=self.min_weight)

            self.task_weights = new_weights / torch.sum(new_weights)

        return self.task_weights.clone()

class TransformSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        img, label, task_type = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label, task_type

    def __len__(self):
        return len(self.indices)


class TaskDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        self.valid_files = []
        for img_file in self.img_files:
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)
            if os.path.exists(label_path):
                self.valid_files.append(img_file)
            else:
                print(f"警告: 未找到标签文件 {label_file}（路径：{label_path}）")

        print(f"在 {os.path.basename(img_dir)} 找到 {len(self.valid_files)} 个有效样本")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        img_name = self.valid_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label_file = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_file)

        with open(label_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        task_type = 'R'
        label_value = 0.0

        if len(lines) >= 1:
            try:
                label_value = float(lines[0])
            except ValueError:
                print(f"错误: 标签文件 {label_file} 第一行格式无效（应为数字）")

        if len(lines) >= 2:
            task_type = lines[1].upper()
            if task_type not in ['R', 'C']:
                print(f"警告: 标签文件 {label_file} 第二行无效（应为R/C），默认按回归处理")
                task_type = 'R'

        if task_type == 'C':
            label = int(round(label_value))
            label = max(0, min(label, Config.num_classes - 1))
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = torch.tensor(label_value, dtype=torch.float32)

        return image, label, task_type


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(Config.img_size, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((Config.img_size, Config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


cla_full_dataset = TaskDataset(
    img_dir=Config.cla_img_dir,
    label_dir=Config.cla_label_dir,
    transform=None
)
reg_full_dataset = TaskDataset(
    img_dir=Config.reg_img_dir,
    label_dir=Config.reg_label_dir,
    transform=None
)

cla_train_size = int(Config.train_ratio * len(cla_full_dataset))
cla_test_size = len(cla_full_dataset) - cla_train_size
cla_train_indices, cla_test_indices = random_split(
    range(len(cla_full_dataset)), [cla_train_size, cla_test_size],
    generator=torch.Generator().manual_seed(42)
)

reg_train_size = int(Config.train_ratio * len(reg_full_dataset))
reg_test_size = len(reg_full_dataset) - reg_train_size
reg_train_indices, reg_test_indices = random_split(
    range(len(reg_full_dataset)), [reg_train_size, reg_test_size],
    generator=torch.Generator().manual_seed(42)
)

cla_train_dataset = TransformSubset(cla_full_dataset, cla_train_indices.indices, train_transform)
cla_test_dataset = TransformSubset(cla_full_dataset, cla_test_indices.indices, test_transform)
reg_train_dataset = TransformSubset(reg_full_dataset, reg_train_indices.indices, train_transform)
reg_test_dataset = TransformSubset(reg_full_dataset, reg_test_indices.indices, test_transform)


cla_train_loader = DataLoader(
    cla_train_dataset,
    batch_size=Config.cla_batch_size,
    shuffle=True,
    drop_last=True
)
reg_train_loader = DataLoader(
    reg_train_dataset,
    batch_size=Config.reg_batch_size,
    shuffle=True,
    drop_last=True
)

cla_test_loader = DataLoader(
    cla_test_dataset,
    batch_size=Config.cla_batch_size,
    shuffle=False,
    drop_last=False
)
reg_test_loader = DataLoader(
    reg_test_dataset,
    batch_size=Config.reg_batch_size,
    shuffle=False,
    drop_last=False
)


print(f"\n训练集 - 分类样本数: {len(cla_train_dataset)}, 回归样本数: {len(reg_train_dataset)}")
print(f"测试集 - 分类样本数: {len(cla_test_dataset)}, 回归样本数: {len(reg_test_dataset)}")
print(f"训练模式: {Config.train_mode}")


class ResNet18MultiTask(nn.Module):
    def __init__(self, config):
        super(ResNet18MultiTask, self).__init__()

        self.resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        backbone_output_dim = self.resnet18.fc.in_features  # 自动获取维度（值为 512）


        self.resnet18.fc = nn.Identity()

        self.freeze_dict = {}
        for name, param in self.resnet18.named_parameters():
            self.freeze_dict[name] = param.requires_grad

        self.regression_head = nn.Sequential(
            nn.Linear(backbone_output_dim, 512),  # 512 → 512（维度适配）
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

        self.classification_head = nn.Sequential(
            nn.Linear(backbone_output_dim, 512),  # 512 → 512（维度适配）
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, Config.num_classes)
        )

    def _freeze_parameters(self):

        if bool(self.freeze_dict) == False:
            print('freeze params must after loading ResNet!')
            os._exit(0)
        freeze_dict = self.freeze_dict
        state_dict = self.state_dict(keep_vars=True)
        for k, v in freeze_dict.items():
            if k in state_dict:
                state_dict[k].requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = True
                m.bias.requires_grad = True
    def forward(self, pixel_values):

        backbone_features = self.resnet18(pixel_values)

        regression_output = self.regression_head(backbone_features).squeeze(dim=1)
        classification_output = self.classification_head(backbone_features)

        return regression_output, classification_output

model = ResNet18MultiTask(Config)
model = model.to(Config.device)

if Config.freeze_backbone:
    model._freeze_parameters()
    print("已冻结 ResNet18 主干网络（除BN层外），仅训练预测头和BN层")

regression_criterion = nn.MSELoss()
classification_criterion = nn.CrossEntropyLoss()


optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)


scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7
)



def train_epoch(model, cla_loader, reg_loader, optimizer, reg_criterion, cls_criterion, device, mode='both', loss_balancer=None):
    model.train()
    total_loss = 0.0
    total_reg_loss = 0.0
    total_cls_loss = 0.0
    total_reg_samples = 0
    total_cls_samples = 0

    if mode == 'classification_only':
        progress_bar = tqdm(cla_loader, desc="训练(仅分类)")
        for cla_images, cla_labels, cla_types in progress_bar:
            cla_images = cla_images.to(device)
            cla_labels = cla_labels.to(device)
            batch_size = cla_images.size(0)

            _, cls_outputs = model(cla_images)

            cls_loss_avg = cls_criterion(cls_outputs, cla_labels)
            cls_loss_avg=cls_loss_avg

            cls_loss_total = cls_loss_avg * batch_size

            optimizer.zero_grad()
            cls_loss_avg.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += cls_loss_total.item()
            total_cls_loss += cls_loss_total.item()
            total_cls_samples += batch_size

            progress_bar.set_postfix(loss=cls_loss_avg.item())

        avg_loss = total_loss / total_cls_samples if total_cls_samples > 0 else 0
        avg_cls_loss = total_cls_loss / total_cls_samples if total_cls_samples > 0 else 0
        return avg_loss, 0.0, avg_cls_loss

    elif mode == 'regression_only':
        progress_bar = tqdm(reg_loader, desc="训练(仅回归)")
        for reg_images, reg_labels, reg_types in progress_bar:
            reg_images = reg_images.to(device)
            reg_labels = reg_labels.to(device)
            batch_size = reg_images.size(0)

            reg_outputs, _ = model(reg_images)

            reg_loss_avg = reg_criterion(reg_outputs, reg_labels)

            reg_loss_total = reg_loss_avg * batch_size

            optimizer.zero_grad()
            reg_loss_avg.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


            total_loss += reg_loss_total.item()
            total_reg_loss += reg_loss_total.item()
            total_reg_samples += batch_size

            progress_bar.set_postfix(loss=reg_loss_avg.item())


        avg_loss = total_loss / total_reg_samples if total_reg_samples > 0 else 0
        avg_reg_loss = total_reg_loss / total_reg_samples if total_reg_samples > 0 else 0
        return avg_loss, avg_reg_loss, 0.0

    else:
        max_batches = max(len(cla_loader), len(reg_loader))
        cla_iter = iter(cla_loader)
        reg_iter = iter(reg_loader)

        progress_bar = tqdm(range(max_batches), desc="训练(多任务)")

        for _ in progress_bar:
            try:
                cla_images, cla_labels, cla_types = next(cla_iter)
            except StopIteration:
                cla_iter = iter(cla_loader)
                cla_images, cla_labels, cla_types = next(cla_iter)

            try:
                reg_images, reg_labels, reg_types = next(reg_iter)
            except StopIteration:
                reg_iter = iter(reg_loader)
                reg_images, reg_labels, reg_types = next(reg_iter)

            images = torch.cat([cla_images, reg_images], dim=0).to(device)
            cla_labels = cla_labels.to(device)
            reg_labels = reg_labels.to(device)
            cla_batch_size = len(cla_types)
            reg_batch_size = len(reg_types)

            reg_outputs, cls_outputs = model(images)

            cls_loss_sum = 0.0
            for i in range(cla_batch_size):
                loss = cls_criterion(cls_outputs[i:i + 1], cla_labels[i:i + 1])
                cls_loss_sum += loss
            cls_avg_loss = cls_loss_sum / cla_batch_size

            reg_loss_sum = 0.0
            for i in range(reg_batch_size):
                idx = cla_batch_size + i
                loss = reg_criterion(reg_outputs[idx:idx + 1], reg_labels[i:i + 1])
                reg_loss_sum += loss
            reg_avg_loss = reg_loss_sum / reg_batch_size

            current_losses = [cls_avg_loss.item(), reg_avg_loss.item()]
            task_weights = loss_balancer.update(current_losses, device)
            cls_weight, reg_weight = task_weights[0], task_weights[1]

            batch_loss = cls_weight * cls_avg_loss + reg_weight * reg_avg_loss

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += batch_loss.item() * (cla_batch_size + reg_batch_size)
            total_cls_loss += cls_loss_sum.item()
            total_reg_loss += reg_loss_sum.item()
            total_cls_samples += cla_batch_size
            total_reg_samples += reg_batch_size

            progress_bar.set_postfix(loss=batch_loss.item())


        avg_loss = total_loss / (total_cls_samples + total_reg_samples) if (
                                                                                       total_cls_samples + total_reg_samples) > 0 else 0
        avg_cls_loss = total_cls_loss / total_cls_samples if total_cls_samples > 0 else 0
        avg_reg_loss = total_reg_loss / total_reg_samples if total_reg_samples > 0 else 0

        return avg_loss, avg_reg_loss, avg_cls_loss


def evaluate(model, cla_loader, reg_loader, reg_criterion, cls_criterion, device, mode='both',loss_balancer=None):
    model.eval()
    total_loss = 0.0
    total_reg_loss = 0.0
    total_cls_loss = 0.0
    total_reg_samples = 0
    total_cls_samples = 0

    all_reg_outputs = []
    all_reg_labels = []
    all_cls_outputs = []
    all_cls_labels = []

    with torch.no_grad():
        if mode == 'classification_only' or mode == 'both':
            for images, labels, task_types in tqdm(cla_loader, desc="评估分类数据"):
                images = images.to(device)
                labels = labels.to(device)
                batch_size = images.size(0)

                reg_outputs, cls_outputs = model(images)

                cls_loss_sum = 0.0
                cls_count = 0
                for i in range(batch_size):
                    if task_types[i] == 'C':
                        loss = cls_criterion(cls_outputs[i:i + 1], labels[i:i + 1])
                        cls_loss_sum += loss
                        cls_count += 1
                        all_cls_outputs.append(torch.argmax(cls_outputs[i]).cpu().item())
                        all_cls_labels.append(labels[i].cpu().item())

                if cls_count > 0:
                    cls_avg_loss = cls_loss_sum / cls_count
                    total_cls_loss += cls_loss_sum.item()
                    total_cls_samples += cls_count
                    if mode == 'both':
                        cls_weight = loss_balancer.task_weights[0]
                        total_loss += cls_weight * cls_avg_loss.item() * cls_count
                    else:
                        total_loss += cls_avg_loss.item() * cls_count
                else:
                    cls_avg_loss = torch.tensor(0.0, device=device).item()  # 关键修改：指定设备+转浮点数

        if mode == 'regression_only' or mode == 'both':
            for images, labels, task_types in tqdm(reg_loader, desc="评估回归数据"):
                images = images.to(device)
                labels = labels.to(device)
                batch_size = images.size(0)

                reg_outputs, cls_outputs = model(images)

                reg_loss_sum = 0.0
                reg_count = 0
                for i in range(batch_size):
                    if task_types[i] == 'R':
                        loss = reg_criterion(reg_outputs[i:i + 1], labels[i:i + 1])
                        reg_loss_sum += loss
                        reg_count += 1
                        all_reg_outputs.append(reg_outputs[i].cpu().item())
                        all_reg_labels.append(labels[i].cpu().item())

                if reg_count > 0:
                    reg_avg_loss = reg_loss_sum / reg_count
                    total_reg_loss += reg_loss_sum.item()
                    total_reg_samples += reg_count
                    if mode == 'both':
                        reg_weight = loss_balancer.task_weights[1]
                        total_loss += reg_weight * reg_avg_loss.item() * reg_count
                    else:
                        total_loss += reg_avg_loss.item() * reg_count
                else:

                    reg_avg_loss = torch.tensor(0.0, device=device).item()

    if mode == 'both':
        avg_loss = total_loss / (total_cls_samples + total_reg_samples) if (
                                                                                       total_cls_samples + total_reg_samples) > 0 else 0
    elif mode == 'classification_only':
        avg_loss = total_cls_loss / total_cls_samples if total_cls_samples > 0 else 0
    elif mode == 'regression_only':
        avg_loss = total_reg_loss / total_reg_samples if total_reg_samples > 0 else 0

    avg_cls_loss = total_cls_loss / total_cls_samples if total_cls_samples > 0 else 0
    avg_reg_loss = total_reg_loss / total_reg_samples if total_reg_samples > 0 else 0

    return (avg_loss, avg_reg_loss, avg_cls_loss,
            all_reg_outputs, all_reg_labels,
            all_cls_outputs, all_cls_labels)


metrics_list = []
train_losses = []
val_losses = []
train_reg_losses = []
train_cls_losses = []
val_reg_losses = []
val_cls_losses = []
best_val_loss = float('inf')
early_stop_patience = 20
early_stop_counter = 0
loss_balancer = None
if Config.train_mode == 'both':

    loss_balancer = DualFactorLossBalancer(num_tasks=2)
print(f"开始训练，使用设备: {Config.device}")
print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

for epoch in range(Config.epochs):
    print(f"\nEpoch {epoch + 1}/{Config.epochs}")

    if (epoch == Config.freeze_epochs) and Config.freeze_backbone:

        for param in model.resnet18.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate / 10, weight_decay=Config.weight_decay)
        print(f"解冻 ResNet18 主干网络（第{epoch + 1}轮），开始全量微调，学习率调整为{Config.learning_rate / 10:.2e}")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )

        Config.freeze_backbone = False

    if Config.train_mode == 'both':

        train_loss, train_reg_loss, train_cls_loss = train_epoch(
            model, cla_train_loader, reg_train_loader, optimizer,
            regression_criterion, classification_criterion,
            Config.device, mode=Config.train_mode,
            loss_balancer=loss_balancer
        )
    else:

        train_loss, train_reg_loss, train_cls_loss = train_epoch(
            model, cla_train_loader, reg_train_loader, optimizer,
            regression_criterion, classification_criterion,
            Config.device, mode=Config.train_mode
        )



    eval_mode = 'both' if Config.train_mode == 'both' else Config.train_mode
    if eval_mode == 'both':

        val_loss, val_reg_loss, val_cls_loss, _, _, _, _ = evaluate(
            model, cla_test_loader, reg_test_loader,
            regression_criterion, classification_criterion,
            Config.device, mode=eval_mode,
            loss_balancer=loss_balancer
        )
    else:

        val_loss, val_reg_loss, val_cls_loss, _, _, _, _ = evaluate(
            model, cla_test_loader, reg_test_loader,
            regression_criterion, classification_criterion,
            Config.device, mode=eval_mode
        )


    if Config.train_mode == 'both' and loss_balancer is not None:
        loss_balancer.update_validation_losses(
            val_cls_loss=val_cls_loss,
            val_reg_loss=val_reg_loss,
            device=Config.device
        )

        if len(loss_balancer.val_loss_history) >= loss_balancer.min_trend_samples:
            overfitting_tasks = loss_balancer._detect_overfitting()
            overfit_cls = overfitting_tasks[0].item() == 1.0
            overfit_reg = overfitting_tasks[1].item() == 1.0
            print(f"过拟合检测 - 分类任务: {'是' if overfit_cls else '否'}, 回归任务: {'是' if overfit_reg else '否'}")

    cls_weight = None
    reg_weight = None
    if Config.train_mode == 'both' and loss_balancer is not None:
        cls_weight = loss_balancer.task_weights[0].item()  # 分类权重
        reg_weight = loss_balancer.task_weights[1].item()  # 回归权重
        print(f"当前任务权重 - 分类: {cls_weight:.4f}, 回归: {reg_weight:.4f}")
    else:
        cls_weight = np.nan
        reg_weight = np.nan


    train_losses.append(float(train_loss))
    train_reg_losses.append(float(train_reg_loss))
    train_cls_losses.append(float(train_cls_loss))
    val_losses.append(float(val_loss))
    val_reg_losses.append(float(val_reg_loss))
    val_cls_losses.append(float(val_cls_loss))

    current_lr = optimizer.param_groups[0]['lr']


    print(f"训练总损失: {train_loss:.4f}, 训练回归损失: {train_reg_loss:.4f}, 训练分类损失: {train_cls_loss:.4f}")
    print(f"验证总损失: {val_loss:.4f}, 验证回归损失: {val_reg_loss:.4f}, 验证分类损失: {val_cls_loss:.4f}")
    print(f"当前学习率: {current_lr:.2e}")

    current_metrics = {
        'Epoch': epoch + 1,
        'Train_Total_Loss': round(train_loss, 4),
        'Train_Regression_Loss': round(train_reg_loss, 4),
        'Train_Classification_Loss': round(train_cls_loss, 4),
        'Val_Total_Loss': round(float(val_loss), 4),
        'Val_Regression_Loss': round(float(val_reg_loss), 4),
        'Val_Classification_Loss': round(float(val_cls_loss), 4),
        'Classification_Weight': round(cls_weight, 4) if not np.isnan(cls_weight) else np.nan,
        'Regression_Weight': round(reg_weight, 4) if not np.isnan(reg_weight) else np.nan,
        'Learning_Rate': round(current_lr, 8),
        'Is_Backbone_Frozen': Config.freeze_backbone
    }

    metrics_list.append(current_metrics)

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), Config.model_save_path)
        print(f"保存最佳模型到 {Config.model_save_path}")
        early_stop_counter = 0
    else:

        if not ((epoch < Config.freeze_epochs) and Config.freeze_backbone):
            early_stop_counter += 1
            print(f"早停计数器: {early_stop_counter}/{early_stop_patience}")


            if early_stop_counter >= early_stop_patience:
                print(f"验证损失连续{early_stop_patience}轮未下降，提前停止训练")
                break
        else:

            print(f"冻结阶段，验证损失未下降（不累积早停计数）")

metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv(Config.metric_csv_path, index=False, encoding='utf-8-sig')
print(f"\n训练指标表格已保存到: {Config.metric_csv_path}")
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.plot(train_losses, label='train loss')
plt.plot(val_losses, label='test loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.grid(True)

if Config.train_mode == 'both' or Config.train_mode == 'regression_only':
    plt.figure(figsize=(10, 6))
    plt.plot(train_reg_losses, label='train loss(reg)')
    plt.plot(val_reg_losses, label='test loss(reg)')
    plt.xlabel('Epochs')
    plt.ylabel('Regression Loss')
    plt.title('Regression Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.save_dir, "regression_loss_curve.png"))
    plt.show()

if Config.train_mode == 'both' or Config.train_mode == 'classification_only':
    plt.figure(figsize=(10, 6))
    plt.plot(train_cls_losses, label='train loss(cla)')
    plt.plot(val_cls_losses, label='test loss(cla)')
    plt.xlabel('Epochs')
    plt.ylabel('Classification Loss')
    plt.title('Classification Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.save_dir, "classification_loss_curve.png"))
    plt.show()

print("\n最终评估:")
model.load_state_dict(torch.load(Config.model_save_path))

eval_mode = 'both' if Config.train_mode == 'both' else Config.train_mode

if eval_mode == 'both':
    _, _, _, reg_outputs, reg_labels, cls_outputs, cls_labels = evaluate(
        model, cla_test_loader, reg_test_loader,
        regression_criterion, classification_criterion,
        Config.device, mode=eval_mode,
        loss_balancer=loss_balancer
    )
else:
    _, _, _, reg_outputs, reg_labels, cls_outputs, cls_labels = evaluate(
        model, cla_test_loader, reg_test_loader,
        regression_criterion, classification_criterion,
        Config.device, mode=eval_mode
    )


if (Config.train_mode == 'both' or Config.train_mode == 'regression_only') and len(reg_outputs) > 0 and len(
        reg_labels) > 0:
    reg_outputs = np.array(reg_outputs)
    reg_labels = np.array(reg_labels)
    mse = np.mean((reg_outputs - reg_labels) ** 2)
    mae = np.mean(np.abs(reg_outputs - reg_labels))
    r2 = 1 - np.sum((reg_labels - reg_outputs) ** 2) / np.sum((reg_labels - np.mean(reg_labels)) ** 2)

    print("\n回归任务评估:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")


    regression_results_df = pd.DataFrame({
        'True_Value': reg_labels,
        'Predicted_Value': reg_outputs,
        'Absolute_Error': np.abs(reg_outputs - reg_labels),
        'Relative_Error': np.abs((reg_outputs - reg_labels) / (reg_labels + 1e-8))
    })


    excel_save_path = os.path.join(Config.save_dir, "regression_predictions.xlsx")
    regression_results_df.to_excel(excel_save_path, index=False, sheet_name='Regression_Results')

    stats_data = {
        'Metric': ['MSE', 'MAE', 'R²', '样本数量'],
        'Value': [mse, mae, r2, len(reg_labels)]
    }
    stats_df = pd.DataFrame(stats_data)

    with pd.ExcelWriter(excel_save_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        stats_df.to_excel(writer, index=False, sheet_name='Statistics')

    print(f"回归预测结果已保存到: {excel_save_path}")

    reg_metrics_path = os.path.join(Config.save_dir, "resut.txt")


    plt.figure(figsize=(10, 8))
    plt.scatter(reg_labels, reg_outputs, alpha=0.5)
    plt.plot([min(reg_labels), max(reg_labels)], [min(reg_labels), max(reg_labels)], 'r--')
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title('Regression: Predicted vs True Value')
    plt.grid(True)
    plt.savefig(os.path.join(Config.save_dir, "regression_predictions.png"))
    plt.show()

if (Config.train_mode == 'both' or Config.train_mode == 'classification_only') and len(cls_outputs) > 0 and len(
        cls_labels) > 0:
    cls_outputs = np.array(cls_outputs)
    cls_labels = np.array(cls_labels)
    accuracy = np.mean(cls_outputs == cls_labels)

    class_correct = np.zeros(Config.num_classes)
    class_total = np.zeros(Config.num_classes)

    for label, pred in zip(cls_labels, cls_outputs):
        class_correct[label] += (label == pred)
        class_total[label] += 1

    class_accuracy = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                      for i in range(Config.num_classes)]

    print("\n分类任务评估:")
    print(f"总体准确率: {accuracy:.4f}")

    with open(reg_metrics_path, 'w', encoding='utf-8') as f:
        f.write("回归任务评估指标\n")
        f.write("==================\n")
        f.write(f"均方误差 (MSE): {mse:.4f}\n")
        f.write(f"平均绝对误差 (MAE): {mae:.4f}\n")
        f.write(f"决定系数 (R²): {r2:.4f}\n")
        f.write("\n分类任务评估:")
        f.write("==================\n")
        f.write(f"总体准确率: {accuracy:.4f}")
        for i in range(Config.num_classes):
            print(f"类别 {i} 准确率: {class_accuracy[i]:.4f} ({int(class_correct[i])}/{int(class_total[i])})")
            f.write(f"类别 {i} 准确率: {class_accuracy[i]:.4f} ({int(class_correct[i])}/{int(class_total[i])})")

    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(cls_labels, cls_outputs)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Class {i}' for i in range(Config.num_classes)],
                yticklabels=[f'Class {i}' for i in range(Config.num_classes)])
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Classification Confusion Matrix')
    plt.savefig(os.path.join(Config.save_dir, "classification_confusion_matrix.png"))
    plt.show()

print("训练和评估完成!")