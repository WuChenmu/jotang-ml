import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# =========================
# 1. 生成数据并划分
# =========================
X, y = make_moons(n_samples=2000, noise=0.2, random_state=42)
X = X.astype(np.float32)
y = y.astype(np.float32)

# 先分 train+val 和 test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 再分 train 和 val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)
# 最终比例：train 60%，val 20%，test 20%

print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")

# 可视化原始数据
plt.figure(figsize=(6, 5))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', s=20)
plt.title("make_moons 原始数据（训练集）")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# 转为 Tensor
def to_tensor(X, y):
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

X_train_t, y_train_t = to_tensor(X_train, y_train)
X_val_t, y_val_t = to_tensor(X_val, y_val)
X_test_t, y_test_t = to_tensor(X_test, y_test)

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# =========================
# 2. 定义 MLP
# =========================
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, activation='relu'):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.01))
            else:
                raise ValueError(f"未知激活函数: {activation}")
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))  # 二分类，输出一个 logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)  # (B,)

# =========================
# 3. 训练与评估函数
# =========================
def train_model(config, data, device):
    X_train, y_train, X_val, y_val, X_test, y_test = data

    model = MLP(
        input_dim=2,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        activation=config['activation']
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    else:
        raise ValueError(f"未知优化器: {config['optimizer']}")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=256,
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=256,
        shuffle=False
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    best_val_acc = 0.0
    best_model_state = None

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()

    for epoch in range(config['epochs']):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == yb).sum().item()
            train_total += xb.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else 0.0

    # 加载最佳模型
    model.load_state_dict(best_model_state)
    model.to(device)

    # 测试
    model.eval()
    test_probs = []
    test_preds = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            test_probs.append(probs.cpu())
            test_preds.append(preds.cpu())

    test_probs = torch.cat(test_probs).numpy()
    test_preds = torch.cat(test_preds).numpy()
    test_true = y_test.numpy()

    cm = confusion_matrix(test_true.astype(int), test_preds.astype(int))
    test_acc = accuracy_score(test_true.astype(int), test_preds.astype(int))

    metrics = {
        'train_time': train_time,
        'peak_mem_mb': peak_mem,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'confusion_matrix': cm,
        'test_probs': test_probs,
        'test_preds': test_preds,
        'test_true': test_true
    }
    return model, history, metrics

# =========================
# 4. 绘图函数
# =========================
def plot_history(history, name, axs):
    axs[0].plot(history['train_loss'], label='train_loss')
    axs[0].plot(history['val_loss'], label='val_loss')
    axs[0].set_title(f'{name} Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].legend()

    axs[1].plot(history['train_acc'], label='train_acc')
    axs[1].plot(history['val_acc'], label='val_acc')
    axs[1].set_title(f'{name} Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()

def plot_decision_boundary(model, X, y, ax, title, device):
    model.eval()
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(grid)
        probs = torch.sigmoid(logits).cpu().numpy()
    Z = probs.reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, colors=['#FFAAAA', '#AAAAFF'])
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=20)
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

# =========================
# 5. 对照实验配置
# =========================
configs = {
    'base': {
        'hidden_dim': 64, 'num_layers': 2, 'activation': 'relu',
        'lr': 0.01, 'optimizer': 'adam', 'batch_size': 64, 'epochs': 100
    },
    'wider': {  # 只改变隐藏层宽度
        'hidden_dim': 128, 'num_layers': 2, 'activation': 'relu',
        'lr': 0.01, 'optimizer': 'adam', 'batch_size': 64, 'epochs': 100
    },
    'lower_lr': {  # 只改变学习率
        'hidden_dim': 64, 'num_layers': 2, 'activation': 'relu',
        'lr': 0.001, 'optimizer': 'adam', 'batch_size': 64, 'epochs': 100
    }
}

data = (X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t)
results = {}
models = {}

for name, config in configs.items():
    print(f"\n===== 训练配置: {name} =====")
    model, history, metrics = train_model(config, data, device)
    models[name] = model
    results[name] = metrics

    # 保存模型
    torch.save(model.state_dict(), f'model_{name}.pt')

    # 绘图
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    plot_history(history, name, axs[:2])
    plot_decision_boundary(model, X_test, y_test, axs[2], f'{name} Decision Boundary', device)
    plt.tight_layout()
    plt.savefig(f'result_{name}.png')
    plt.show()

    print(f"{name} 测试准确率: {metrics['test_acc']:.4f}")
    print(f"{name} 训练时间: {metrics['train_time']:.2f} s")
    print(f"{name} 峰值显存: {metrics['peak_mem_mb']:.2f} MB")
    print("混淆矩阵:")
    print(metrics['confusion_matrix'])

# =========================
# 6. 汇总结果
# =========================
print("\n===== 对照实验汇总 =====")
print(f"{'配置':<10} {'测试准确率':<10} {'训练时间(s)':<12} {'峰值显存(MB)':<15}")
for name, metrics in results.items():
    print(f"{name:<10} {metrics['test_acc']:<10.4f} {metrics['train_time']:<12.2f} {metrics['peak_mem_mb']:<15.2f}")

# =========================
# 7. 错误样本分析（以 base 为例）
# =========================
base_metrics = results['base']
test_true = base_metrics['test_true']
test_preds = base_metrics['test_preds']
test_probs = base_metrics['test_probs']

wrong_idx = np.where(test_preds != test_true)[0]
print(f"\n基模型错误样本数: {len(wrong_idx)} / {len(test_true)}")

for i in wrong_idx[:5]:
    print(f"索引 {i}: 坐标 {X_test[i]}, 真实标签 {test_true[i]}, "
          f"预测概率 {test_probs[i]:.4f}, 预测标签 {test_preds[i]}")

# =========================
# 8. 加载模型示例
# =========================
loaded_model = MLP(input_dim=2, hidden_dim=64, num_layers=2, activation='relu').to(device)
loaded_model.load_state_dict(torch.load('model_base.pt', map_location=device))
loaded_model.eval()
print("\n已成功加载 model_base.pt")