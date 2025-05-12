import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
import matplotlib.pyplot as plt
from PIL import Image
import os 
import pathlib 
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
plt.rcParams['font.sans-serif']=['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 定义SE注意力模块
def se_block(input_tensor, ratio=4):
    """ Squeeze-and-Excitation块 """
    channels = input_tensor.shape[-1]
    
    # Squeeze操作（全局平均池化）
    x = layers.GlobalAveragePooling2D()(input_tensor)
    
    # Excitation操作（两个全连接层）
    x = layers.Dense(channels//ratio, activation='silu')(x)
    x = layers.Dense(channels, activation='sigmoid')(x)
    
    # 调整维度用于乘法操作
    x = layers.Reshape((1, 1, channels))(x)
    
    # 通道权重调整
    return layers.Multiply()([input_tensor, x])

# 数据路径设置
path = 'Dataset_Images'
data_dir = pathlib.Path(path)

# 数据集统计
benignPath = pathlib.Path(os.path.join(data_dir,'benign'))
normalPath = pathlib.Path(os.path.join(data_dir,'normal'))
malignantPath = pathlib.Path(os.path.join(data_dir,'malignant'))

benignImageCount = len(list(benignPath.glob('*.png')))
malignantImageCount = len(list(malignantPath.glob('*.png')))
normalImageCount = len(list(normalPath.glob('*.png')))
totalImageCount = benignImageCount + malignantImageCount + normalImageCount

print("Total number of Images: ", totalImageCount)
print("No. of Benign Images: {}({})".format(benignImageCount, round(benignImageCount*100/totalImageCount, 2)))
print("No. of Malignant Images: {}({})".format(malignantImageCount, round(malignantImageCount*100/totalImageCount, 2)))
print("No. of Normal Images: {}({})".format(normalImageCount, round(normalImageCount*100/totalImageCount, 2)))

# 模型参数
batch_size = 32
img_height = 224
img_width = 224

# 数据加载
train_data = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.25,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_data = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.25,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# 使用函数式API构建模型
inputs = layers.Input(shape=(img_height, img_width, 3))
x = layers.Rescaling(1./255)(inputs)

# 卷积块1
x = layers.Conv2D(4, 3, padding='same', activation='leaky_relu')(x)
x = se_block(x)  # 添加SE模块
x = layers.MaxPooling2D()(x)

# 卷积块2
x = layers.Conv2D(8, 3, padding='same', activation='leaky_relu')(x)
x = se_block(x)  # 添加SE模块
x = layers.MaxPooling2D()(x)

# 卷积块3
x = layers.Conv2D(16, 3, padding='same', activation='leaky_relu')(x)
x = se_block(x)  # 添加SE模块
x = layers.MaxPooling2D()(x)

# 卷积块4
x = layers.Conv2D(32, 3, padding='same', activation='leaky_relu')(x)
x = se_block(x)  # 添加SE模块
x = layers.MaxPooling2D()(x)

# 卷积块5
x = layers.Conv2D(64, 3, padding='same', activation='leaky_relu')(x)
x = se_block(x)  # 添加SE模块
x = layers.MaxPooling2D()(x)

# 卷积块6
x = layers.Conv2D(128, 3, padding='same', activation='leaky_relu')(x)
x = se_block(x)  # 添加SE模块
x = layers.MaxPooling2D()(x)

# 分类层
x = layers.Dropout(0.9)(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation='leaky_relu')(x)
outputs = layers.Dense(3, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# 编译模型
model.compile(optimizer="Adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])

# 训练模型
epochs = 50
history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=val_data, 
    batch_size=batch_size)

y_true = []
y_pred = []

# 遍历整个验证数据集
for images, labels in val_data:
    # 批量预测
    predictions = model.predict(images, verbose=0)
    # 收集真实标签
    y_true.extend(labels.numpy())
    # 收集预测标签（取概率最大的类别）
    y_pred.extend(np.argmax(predictions, axis=1))

# 生成混淆矩阵
cm = confusion_matrix(y_true, y_pred)
class_names = val_data.class_names

# 可视化设置
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel('预测类别', fontsize=12)
plt.ylabel('真实类别', fontsize=12)
plt.title('混淆矩阵分析', fontsize=15)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
# 可视化训练过程
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='训练准确率')
plt.plot(epochs_range, val_acc, label='验证准确率')
plt.legend(loc='lower right')
plt.title('训练和验证准确率')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='训练损失')
plt.plot(epochs_range, val_loss, label='验证损失')
plt.legend(loc='upper right')
plt.title('训练和验证损失')

plt.show()

# 模型评估
val_loss, val_acc = model.evaluate(val_data)
print(f'验证集损失: {val_loss}')
print(f'验证集准确率: {val_acc}')

# 预测可视化
plt.figure(figsize=(15, 15))
class_names = val_data.class_names
plt.subplots_adjust(hspace=0.5, wspace=0.5)

for images, labels in val_data.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        img = images[i].numpy().astype("uint8")
        img = tf.expand_dims(img, axis=0)

        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)

        true_class = labels[i].numpy()
        true_class_name = class_names[true_class]
        predicted_class_name = class_names[predicted_class]

        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"True: {true_class_name}\nPred: {predicted_class_name}")
        plt.axis("off")

plt.show()