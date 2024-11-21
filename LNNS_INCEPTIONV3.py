import os
import shutil
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, RNN, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


# --------------------------
# 数据自动划分
# --------------------------

def split_data(input_dir, output_dir, split_ratio=0.8):
    """
    将单一文件夹中的数据按比例划分为训练集和验证集。
    :param input_dir: 所有图片存放的文件夹路径，按子文件夹分类。
    :param output_dir: 输出的训练和验证数据集文件夹路径。
    :param split_ratio: 训练数据的比例（默认 80%）。
    """
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if not os.path.isdir(category_path):
            continue

        # 获取该类别下的所有图片路径
        images = [os.path.join(category_path, img) for img in os.listdir(category_path) if
                  img.endswith(('.jpg', '.png'))]

        # 划分训练和验证
        train_images, val_images = train_test_split(images, train_size=split_ratio, random_state=42)

        # 创建类别文件夹
        train_category_path = os.path.join(train_dir, category)
        val_category_path = os.path.join(val_dir, category)
        os.makedirs(train_category_path, exist_ok=True)
        os.makedirs(val_category_path, exist_ok=True)

        # 移动文件
        for img_path in train_images:
            shutil.copy(img_path, train_category_path)
        for img_path in val_images:
            shutil.copy(img_path, val_category_path)

    print(f"数据划分完成，训练集存放于：{train_dir}，验证集存放于：{val_dir}")
    return train_dir, val_dir


# --------------------------
# 自定义 LNN 单元
# --------------------------

class LiquidNeuralCell(Layer):
    def __init__(self, units, **kwargs):
        super(LiquidNeuralCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units), initializer="glorot_uniform", trainable=True, name="kernel"
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units), initializer="orthogonal", trainable=True, name="recurrent_kernel"
        )
        self.bias = self.add_weight(
            shape=(self.units,), initializer="zeros", trainable=True, name="bias"
        )

    def call(self, inputs, states):
        prev_output = states[0]
        h = tf.matmul(inputs, self.kernel) + tf.matmul(prev_output, self.recurrent_kernel) + self.bias
        output = tf.nn.tanh(h)
        return output, [output]

    def get_config(self):
        config = super(LiquidNeuralCell, self).get_config()
        config.update({"units": self.units})
        return config




# --------------------------
# 模型构建
# --------------------------

def create_model(input_shape, num_classes):
    # 加载 Inception v3 作为特征提取器
    inception_base = InceptionV3(weights="imagenet", include_top=False, input_shape=input_shape)
    inception_base.trainable = False  # 冻结卷积层

    # 定义模型结构
    input_layer = Input(shape=input_shape)
    x = inception_base(input_layer)
    x = GlobalAveragePooling2D()(x)

    # LNN 模块
    lnn_cell = LiquidNeuralCell(units=128)
    lnn_layer = RNN(lnn_cell)(tf.expand_dims(x, axis=1))  # 添加时间维度后传入 LNN

    # 分类头
    output_layer = Dense(num_classes, activation="softmax")(lnn_layer)

    # 构建模型
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


# --------------------------
# 主函数
# --------------------------

def main():
    # 原始数据路径
    input_dir = "C:/Users/LA/Documents/BaiduSyncdisk/YYS/tensorflow-yys/images/train_jpg"  # 替换为你的图片文件夹路径（包含各类的子文件夹）
    output_dir = "C:/Users/LA/Documents/BaiduSyncdisk/YYS/tensorflow-yys/split_data"  # 划分后的数据存放路径

    # 自动划分数据
    train_dir, val_dir = split_data(input_dir, output_dir, split_ratio=0.8)

    # 图像大小与批量大小
    img_size = (224, 224, 3)
    batch_size = 32

    # 数据加载与增强
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size[:2],
        batch_size=batch_size,
        class_mode="categorical"
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size[:2],
        batch_size=batch_size,
        class_mode="categorical"
    )

    # 创建模型
    num_classes = len(train_generator.class_indices)
    model = create_model(img_size, num_classes)

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 训练模型
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        steps_per_epoch=len(train_generator),
        validation_steps=len(val_generator)
    )

    # 保存模型


    model.save("liquid_nn_inception_model.h5", save_format="h5")


    print("Model saved successfully.")


if __name__ == "__main__":
    main()
