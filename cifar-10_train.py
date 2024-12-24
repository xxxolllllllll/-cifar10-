import os
from pickletools import optimize

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils.readData import read_dataset, device, batch_size
from utils.ResNet import resnet18
from utils.ResNet_opt import resnet18_opt
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score



# 模型的训练
def train(epoch, optimizer):
    # keep track of training and validation loss
    train_loss = 0.0
    total_sample = 0
    right_sample = 0
    all_preds = []
    all_targets = []

    # # 动态调整学习率
    # if counter / 10 == 1:
    #     counter = 0
    #     lr = lr * 0.5
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    ###################
    # 训练集的模型 #
    ###################
    model.train()  # 作用是启用batch normalization和drop out
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        # clear the gradients of all optimized variables（清除梯度）
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        # (正向传递：通过向模型传递输入来计算预测输出)
        output = model(data).to(device)  # （等价于output = model.forward(data).to(device) ）
        # calculate the batch loss（计算损失值）
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        # （反向传递：计算损失相对于模型参数的梯度）
        loss.backward()
        # perform a single optimization step (parameter update)
        # 执行单个优化步骤（参数更新）
        optimizer.step()
        # update training loss（更新损失）
        train_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class(将输出概率转换为预测类)
        _, pred = torch.max(output, 1)
        # compare predictions to true label(将预测与真实标签进行比较)
        correct_tensor = pred.eq(target.data.view_as(pred))
        # correct = np.squeeze(correct_tensor.to(device).numpy())
        total_sample += batch_size
        for i in correct_tensor:
            if i:
                right_sample += 1
        # 保存预测和真实标签用于后续计算精度、召回率
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

    acc = 100 * right_sample / total_sample
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=1)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=1)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=1)
    print(f"Train Epoch: {epoch} | Loss: {train_loss / len(train_loader):.4f} | "
          f"Acc: {acc:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    # 将指标保存到 TensorBoard
    writer.add_scalar('log/train loss', train_loss, epoch)
    writer.add_scalar('log/train error', 100 - acc, epoch)
    writer.add_scalar('log/train precision', precision, epoch)
    writer.add_scalar('log/train recall', recall, epoch)
    writer.add_scalar('log/train f1', f1, epoch)

    return {
        'epoch': epoch,
        'train_loss': train_loss / len(train_loader),
        'train_acc': acc,
        'train_precision': precision,
        'train_recall': recall,
        'train_f1': f1
    }

# 模型的验证
def valid(epoch, best_acc):
    valid_loss = 0.0
    total_sample = 0
    right_sample = 0
    all_preds = []
    all_targets = []
    counter = 0
    ######################
    # 验证集的模型#
    ######################

    model.eval()  # 验证模型
    for data, target in valid_loader:
        data = data.to(device)
        target = target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data).to(device)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss
        valid_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class(将输出概率转换为预测类)
        _, pred = torch.max(output, 1)
        # compare predictions to true label(将预测与真实标签进行比较)
        correct_tensor = pred.eq(target.data.view_as(pred))
        # correct = np.squeeze(correct_tensor.to(device).numpy())
        total_sample += batch_size
        for i in correct_tensor:
            if i:
                right_sample += 1
        # 保存预测和真实标签用于后续计算精度、召回率
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

    acc = 100 * right_sample / total_sample
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=1)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=1)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=1)
    print(f"Valid Epoch: {epoch} | Loss: {valid_loss / len(valid_loader):.4f} | "
          f"Acc: {acc:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    # 将指标保存到 TensorBoard
    writer.add_scalar('log/valid loss', valid_loss, epoch)
    writer.add_scalar('log/valid error', 100 - acc, epoch)
    writer.add_scalar('log/valid precision', precision, epoch)
    writer.add_scalar('log/valid recall', recall, epoch)
    writer.add_scalar('log/valid f1', f1, epoch)

    # 如果测试精度更好，保存模型
    if acc > best_acc:
        print('==> Saving model..')
        if not os.path.isdir('../checkpoint'):
            os.mkdir('../checkpoint')
        torch.save(model.state_dict(), '../checkpoint/ckpt_opt_SGDM_lr0.1_epo200_mom0.9.pt')
        best_acc = acc
    print('best valid accuracy is ', best_acc)
    return {
        'epoch': epoch,
        'valid_loss': valid_loss / len(valid_loader),
        'valid_acc': acc,
        'valid_precision': precision,
        'valid_recall': recall,
        'valid_f1': f1
    }, best_acc

def save_metrics_to_excel():
    # 将metrics保存为DataFrame并输出到Excel文件
    df = pd.DataFrame(metrics)
    df.to_excel('../training_metrics.xlsx', index=False)
    print("Metrics saved to training_metrics.xlsx")

if __name__ == '__main__':
    best_acc = 0

    # 用于保存每一轮的指标
    metrics = []

    # 设置gup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 读数据，设置batch_size为64
    batch_size = 64
    train_loader, valid_loader, test_loader = read_dataset(batch_size=batch_size, pic_path='../data')

    # 加载模型(使用预处理模型，修改最后一层，固定之前的权重)
    n_classes = 10
    # model = resnet18(num_classes=n_classes)

    model = resnet18_opt(num_classes=n_classes)

    model = model.to(device)

    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter('../logs_opt/event_SGDM_lr0.1_epo200_mom0.9')
    # 开始训练
    # n_epochs = 150
    n_epochs = 200
    # n_epochs = 250

    lr = 0.1
    # lr = 0.01
    # lr = 0.001

    # SGDM
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=1e-4)

    # SDG
    # optimizer = optim.SGD(model.parameters(), lr=lr , weight_decay=5e-4)

    # nAdam
    # optimizer = optim.NAdam(model.parameters(), lr=lr)

    # Adam
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-4)

    # Adadelta
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)

    # RMSprop
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr,weight_decay=1e-5)

    # print(lr, n_epochs)

    for epoch in tqdm(range(1, n_epochs + 1)):
        train_metrics = train(epoch,optimizer=optimizer)
        valid_metrics, best_acc = valid(epoch, best_acc)
        train_metrics.update(valid_metrics)
        metrics.append(train_metrics)
    writer.close()

    # 保存指标到Excel文件
    save_metrics_to_excel()