
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
import matplotlib.pyplot as plt
from PIL import Image
import os 
import pathlib 
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
plt.rcParams['font.sans-serif']=['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
"""**Defining the path**"""

path = 'Dataset_Images'
data_dir = pathlib.Path(path)

"""**Getting class names**"""

class_names = np.array([item.name for item in data_dir.glob("*")])
class_names

"""**Define paths and image count**"""

benignPath = pathlib.Path(os.path.join(data_dir,'benign'))
normalPath = pathlib.Path(os.path.join(data_dir,'normal'))
malignantPath = pathlib.Path(os.path.join(data_dir,'malignant'))

"""**Image count**"""

benignImageCount = len(list(benignPath.glob('*.png')))
malignantImageCount = len(list(malignantPath.glob('*.png')))
normalImageCount = len(list(normalPath.glob('*.png')))
totalImageCount = benignImageCount + malignantImageCount + normalImageCount

print("Total number of Images: ", totalImageCount)
print("No. of Benign (non-dangerous) Images: {}({})".format(benignImageCount, round(benignImageCount*100/totalImageCount, 2)))
print("No. of Malignant (dangerous) Images: {}({})".format(malignantImageCount, round(malignantImageCount*100/totalImageCount, 2)))
print("No. of Normal (No Traces) Images: {}({})".format(normalImageCount, round(normalImageCount*100/totalImageCount, 2)))

"""# Build the CNN"""

batch_size = 32
img_height = 224
img_width = 224

"""**Separating data sets**"""

from tensorflow.keras.utils import image_dataset_from_directory

train_data = image_dataset_from_directory(data_dir,validation_split=0.25,subset="training",seed=123,image_size=(img_height, img_width),batch_size=batch_size)

val_data = image_dataset_from_directory(data_dir,validation_split=0.25,subset="validation",seed=123,image_size=(img_height,img_width),batch_size=batch_size)

"""# Define the Model

**Roadmap**

##### We rescale images add a Dropout to avoid the overfitting as we have 4 class the last layer contain the number of class and we have softmax as activation,it will give us a pourcentage of each class and we'll choice the maximum pourcentage as the class
"""
# 7 layer CNN Model Architecture with 3 Convolution layer each followed by max pooling layer
# Filter size =3X3 and Activation function = Relu
from tensorflow.keras import layers 
model = tf.keras.Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
    
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  layers.Dropout(0.3),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(3,activation="softmax")
])

"""# Compile the Model"""

model.compile(optimizer="Adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])

epochs = 50
history = model.fit(train_data,
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
"""**Keys**"""

history.history.keys()
# 绘制训练和验证准确率与损失曲线
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
plt.figure(figsize=(12, 6))

# 准确率曲线
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='训练准确率')
plt.plot(epochs_range, val_acc, label='验证准确率')
plt.legend(loc='lower right')
plt.title('训练和验证准确率')

# 损失曲线
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='训练损失')
plt.plot(epochs_range, val_loss, label='验证损失')
plt.legend(loc='upper right')
plt.title('训练和验证损失')

plt.show()

# 评估模型
val_loss, val_acc = model.evaluate(val_data)
print(f'验证集损失: {val_loss}')
print(f'验证集准确率: {val_acc}')
# 测试模型预测
plt.figure(figsize=(15, 15))
class_names = val_data.class_names

# 增加图像间隔
plt.subplots_adjust(hspace=0.5, wspace=0.5)  # 增加水平和垂直间隔

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

        # 在标题中显示真实类别和预测类别
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"True: {true_class_name}\nPred: {predicted_class_name}")
        plt.axis("off")

plt.show()

