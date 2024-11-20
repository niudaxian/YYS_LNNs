import tensorflow as tf
import yys_connect as adb
import time
import numpy as np

# 加载训练好的模型（h5 格式）
model = tf.keras.models.load_model("image_retraining/liquid_nn_image_classifier.h5")

# 加载类别标签
with open("data/retrained_labels_LNNs.txt", "r") as f:
    label_lines = [line.strip() for line in f.readlines()]


# 图像预处理函数
def preprocess_image(image_path, target_size=(224, 224)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # 解码为 RGB 图片
    image = tf.image.resize(image, target_size)  # 缩放到模型输入尺寸
    image = tf.cast(image, tf.float32) / 255.0  # 归一化到 [0, 1]
    return np.expand_dims(image, axis=0)  # 增加批次维度


# 主逻辑
def main():
    # 连接 adb 并显示设备
    adb.connect_adb()
    adb.show_devices()

    while True:
        print("\n----------------------------------------------")
        start = time.time()

        # 截取屏幕并保存为临时图片
        image_path = adb.cap()  # 假设 adb.cap() 返回图像路径

        # 预处理图像
        preprocessed_image = preprocess_image(image_path)

        # 模型预测
        predictions = model.predict(preprocessed_image)
        top_k = np.argsort(predictions[0])[::-1]  # 按概率从大到小排序

        # 获取预测类别和对应标签
        label = label_lines[top_k[0]]
        print(f"预测图片为：{label} ({adb.Labels[label]})")

        # 根据预测结果执行点击动作
        adb.click(label)
        print('耗时: %.3f 秒' % (time.time() - start))


# 执行主逻辑
if __name__ == "__main__":
    main()
