import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
import os

# 数据路径
train_dir = r"C:\Users\LA\Documents\BaiduSyncdisk\YYS\tensorflow-yys\images\train_jpg"  # 替换为你的实际路径
img_height, img_width = 224, 224  # 将图片缩放到 224×224
batch_size = 32

# 获取分类数目
num_classes = len([name for name in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, name))])

# 数据生成器
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,   # 像素值归一化到[0, 1]
    validation_split=0.2 # 将数据划分为训练集和验证集
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),  # 缩放到目标分辨率
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # 训练集
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),  # 缩放到目标分辨率
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # 验证集
)

# 构建简单卷积神经网络模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 模型训练
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=validation_generator.samples // batch_size
)

# 保存模型
model.save("liquid_nn_image_classifier.h5")
print("模型已保存为 'liquid_nn_image_classifier.h5'")
