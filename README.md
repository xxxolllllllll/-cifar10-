

# CIFAR-10 分类与 Resnet18

本项目实现了使用 **Resnet18** 网络结构对 **CIFAR-10** 数据集进行分类的深度学习模型。该模型使用 **PyTorch** 进行训练，并支持将训练过程中的各种指标（如精度、损失、精确度、召回率和 F1 分数）保存到 **TensorBoard** 和 **Excel 文件** 中，便于后续分析。

## 目录结构

```plaintext
.
├── cifar-10_train.py               # 主训练脚本
├── cifar-10_test.py                # 主测试脚本
├── checkpoint/                     # 存储模型检查点
├── data                            # 数据集
├── logs/                           # 存储 TensorBoard 日志
├── training_metrics.xlsx           # 存储训练和测试指标的 Excel 文件
├── requirements.txt                # 依赖包列表
└──  utils                          # 工具包
      ├── readData.py               # 读取数据并进行数据增强
      ├── ResNet.py                 # 模型
      ├── ResNet——opt.py            # 改进模型
      └── cutout.py                 # 在输入的图像上随机“剪切”出一定数量的方形区域，并将这些区域的像素值设置为0（即黑色）
```

## 环境要求

- Python 3.9
- PyTorch 2.4+（支持 CUDA）
- torchvision
- tensorboard
- pandas
- scikit-learn

## 数据集

该项目使用 **CIFAR-10** 数据集，包含 10 个类别的 60,000 张 32x32 彩色图片。数据集会自动下载并存储在 `../data` 目录下。

## 如何训练模型

### 1. 从头开始训练

您可以运行以下命令从头开始训练模型：

```bash
python train.py 
```

这将启动训练过程并在每个 epoch 结束时将训练指标（如损失、精度等）记录到 **TensorBoard**。训练过程中会定期保存模型到 `./checkpoint/`下。

### 2. 从检查点恢复模型

如果您想从之前保存的模型检查点恢复模型，可以在cifar-10_test.py中

```bash
model.load_state_dict(torch.load('model_path'))
```

该命令将从 `./checkpoint/` 下加载模型。
加载模型后，会对测试集进行预测，并返回测试精度。

### 参数说明

- model_path：./checkpoint/下的模型，如'checkpoint/ckpt_nadam_lr0.01_epo250.pt'

## 训练过程

训练过程将输出以下信息：

```plaintext
Train Epoch: <epoch> | Loss: <train_loss> | Acc: <train_acc>% | Precision: <train_precision> | Recall: <train_recall> | F1: <train_f1>
Valid Epoch: <epoch> | Loss: <valid_loss> | Acc: <valid_acc>% | Precision: <valid_precision> | Recall: <valid_recall> | F1: <valid_f1>
```

每个 epoch 的训练和测试结果将被记录到 **TensorBoard**，包括：

- `train error`: 训练误差
- `train precision`: 训练精确度
- `train recall`: 训练召回率
- `train f1`: 训练 F1 分数
- `valid error`: 验证误差
- `valid precision`: 验证精确度
- `valid recall`: 验证召回率
- `valid f1`: 验证 F1 分数

### 查看训练日志

您可以使用 **TensorBoard** 查看训练过程的可视化日志：

```bash
tensorboard --logdir=logs
```

## 模型保存与恢复

- 在训练过程中，每当测试精度（Accuracy）有提升时，模型会保存到 `./checkpoint/` 文件夹中。


## 训练和测试指标保存

训练过程中的所有指标（训练损失、精度、精确度、召回率、F1 分数等）将保存到 `training_metrics.xlsx` 文件中，您可以通过 Excel 进一步分析训练过程。

## 结果

经过训练，模型在测试集上的最佳精度达到了 **93.14%**。


## 代码解析

### 训练过程 (`cifar-10_train.py`)

- **训练过程**：每个 epoch 的训练过程都会计算损失，并根据网络输出与目标的匹配度计算精度。训练结束后，还会计算精确度、召回率和 F1 分数。
- **测试过程**：每个 epoch 后都会进行一次测试，计算测试集上的精度和损失。如果测试精度有所提升，模型会被保存。
- **优化器**：本项目使用 **SGD、SGDM、Adam、NAdam、Adadelta、RMSprop** 等优化器。
- **指标记录与保存**：训练和测试过程中的精度、召回率、F1 等指标被保存到 **TensorBoard** 和 **Excel** 文件中，以便后续查看和分析。

### 主要函数

- **train(epoch)**: 执行训练过程并返回训练指标。
- **valid(epoch, best_acc)**: 执行测试过程，计算并返回测试指标，保存最优模型。
- **save_metrics_to_excel()**: 将训练和测试的指标保存到 `training_metrics.xlsx` 文件中。
- **read_dataset()**:读取数据并进行数据增强

## 许可证

本项目采用 MIT 许可证，详情请参见 [LICENSE](https://chatgpt.com/c/LICENSE) 文件。

