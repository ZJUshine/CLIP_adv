import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from CLIP import clip
from tqdm import tqdm
import os

# 数据集选择
dataset_choice = "CIFAR10"

# 数据集类别
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# 选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 创建文件夹
if not os.path.exists("result"):
    os.makedirs("result")

# 加载模型
model, preprocess = clip.load("ViT-B/32", device=device)
text  = clip.tokenize(class_names).to(device)

# 加载数据集
if dataset_choice == "CIFAR10":
    train_data = datasets.CIFAR10(root='./data', train=True, transform=preprocess, download=True)
    test_data = datasets.CIFAR10(root='./data', train=False, transform=preprocess, download=True)

# 计算CLIP模型的原始准确率
def calculate_accuracy(data):
    correct = 0
    total = 0
    for image, label in tqdm(data):
        image = image.unsqueeze(0).to(device)

        # 模型预测
        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text)

        # 计算预测标签
        prediction = logits_per_image.argmax(1).item()

        # 计算正确个数
        if prediction == label:
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy

# # 计算训练集准确率
# test_accuracy = calculate_accuracy(test_data)
# print(f'Test Accuracy: {test_accuracy:.4f}')

# 冻结模型参数
for p in model.parameters():
  p.requires_grad = False

# 从数据集中选取一张图片
image_original, true_label = test_data[1]
image_original = image_original.unsqueeze(0).to(device)
image_original_np = image_original.detach().cpu().numpy()

# 选择一个目标标签
target_label = 1

print("原始标签:", class_names[true_label])
print("目标标签:", class_names[target_label])

# 创建一个形状为[1,len(class_names)]的张量，其中目标标签的位置为1，其余位置为0
labels = torch.zeros([1,len(class_names)])
labels[0,target_label] = 1
labels = labels.to(device)

# 定义学习率和迭代步数
LR = 0.5 # 学习率
steps = 30 # 迭代步数
criterion = torch.nn.CrossEntropyLoss() # 交叉熵为损失函数

image = image_original.clone()
image_np = image_original_np
# 开始迭代生成对抗样本
for step in range(steps):
  # 解冻图像参数
  image.requires_grad = True
  # 模型预测
  outputs, _ = model(image, text)
  # 计算各个种类的概率
  probs = outputs.softmax(dim=-1)
  # 计算损失
  loss = criterion(probs, labels)

  # 保留梯度并反向传播
  image.retain_grad()
  loss.retain_grad()
  loss.backward()
  loss_out_np = loss.data.detach().cpu().numpy()

  if step % 5 == 0:
    # 打印损失和概率
    print(
      "step="+str(step)+
      " loss="+str(loss_out_np)+
      " p[true="+str(class_names[true_label])+"]="+str(probs.detach().cpu().numpy()[0,true_label])+
      " p[target="+str(class_names[target_label])+"]="+str(probs.detach().cpu().numpy()[0,target_label])
    )
    # 归一化
    image_origin_np_normalized = (image_original_np - image_original_np.min()) / (image_original_np.max() - image_original_np.min())
    image_np_normalized = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    # 保存图片
    plt.figure(figsize=(3*4,4))
    plt.subplot(1,3,1)
    plt.imshow(image_origin_np_normalized[0].clip(0,1).transpose([1,2,0]))
    plt.subplot(1,3,2)
    plt.imshow(image_np_normalized[0].transpose([1,2,0]))
    plt.subplot(1,3,3)
    plt.imshow(np.max(image_np_normalized[0]-image_origin_np_normalized[0],axis=0))
    plt.savefig(f"result/adv_{step}.png")

  image_grad = image.grad.detach().cpu().numpy()
  # 更新图像
  image_np = image.detach().cpu().numpy() - LR*image_grad
  image = torch.Tensor(image_np).to(device)

# 绘制结果函数
def make_plot_from_preds(orig_preds,mod_preds,class_labels,colors = ["navy","crimson"]):

  width = 0.45

  for i in range(orig_preds.shape[0]):
    v_orig = orig_preds[i]
    v_mod = mod_preds[i]

    label1 = ""
    label2 = ""

    alpha_mod = 0.6
    alpha_orig = 0.6

    if np.argmax(mod_preds) == i:
      alpha_mod = 1.0
      label2 = "Adversarial img"
    if np.argmax(orig_preds) == i:
      alpha_orig = 1.0
      label1 = "Original img"

    plt.fill_between([0,v_mod],[i,i],[i+width,i+width], color = colors[1],label = label2, alpha = alpha_mod)
    plt.fill_between([0,v_orig],[i-width,i-width],[i,i], color = colors[0],label = label1, alpha = alpha_orig)

  plt.yticks(range(len(class_labels)),class_labels)

  plt.xticks()

  plt.xlabel("Probability")
  plt.legend()

  plt.xlim([-0.015,1.0])

# 计算原始样本的概率分布
orig_preds = model(image_original, text)[0].softmax(dim=-1).detach().cpu().numpy()
adv_preds = model(image, text)[0].softmax(dim=-1).detach().cpu().numpy()


plt.figure(figsize = (3*4.5,4))
# 第一张图
plt.subplot(1,3,1)
plt.title("original image\n" + class_names[np.argmax(orig_preds[0])],fontsize = 16)
plt.imshow(image_origin_np_normalized[0].transpose([1,2,0]))
plt.grid(False)
plt.xticks([],[])
plt.yticks([],[])
# 第二张图
plt.subplot(1,3,2)
plt.title("Adversarial image\n" + class_names[np.argmax(adv_preds[0])],fontsize = 16)
plt.imshow(image_np_normalized[0].transpose([1,2,0]))
plt.grid(False)
plt.xticks([],[])
plt.yticks([],[])
# 第三张图
plt.subplot(1,3,3)
make_plot_from_preds(orig_preds[0],adv_preds[0],class_names)

plt.tight_layout()
plt.savefig("result/result.png")