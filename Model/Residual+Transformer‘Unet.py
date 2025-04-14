import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.metrics import MeanIoU

# 数据集路径
data_dir = "Data2/压缩/Data3/archive"
# 自定义数据集加载
class RoadDataset(tf.keras.utils.Sequence):
    def __init__(self, folder, batch_size, img_size=(256, 256), shuffle=True):
        self.image_paths = sorted(os.listdir(os.path.join(folder, "image")))
        self.label_paths = sorted(os.listdir(os.path.join(folder, "label")))
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.folder = folder
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, labels = [], []
        for i in batch_indices:
            image_path = os.path.join(self.folder, "image", self.image_paths[i])
            label_path = os.path.join(self.folder, "label", self.label_paths[i])

            # 检查图像文件是否有效
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

            if image is None or label is None:
                print(f"Warning: Skipping invalid image or label file at {image_path} or {label_path}")
                continue  # 跳过无效的图像或标签

            image = cv2.resize(image, self.img_size)
            label = cv2.resize(label, self.img_size)
            images.append(image / 255.0)
            labels.append(label / 255.0)

        # 确保返回的数组不为空
        if len(images) == 0:
            raise ValueError("No valid images in the batch.")
        
        return np.array(images), np.expand_dims(np.array(labels), axis=-1)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

# 定义 Residual Block
def residual_block(x, filters):
    shortcut = layers.Conv2D(filters, (1, 1), padding="same")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([shortcut, x])
    x = layers.ReLU()(x)
    return x

# 定义 Attention Block
def attention_block(x, g, inter_channels):
    theta_x = layers.Conv2D(inter_channels, (1, 1), padding="same")(x)
    phi_g = layers.Conv2D(inter_channels, (1, 1), padding="same")(g)
    f = layers.Activation("relu")(layers.add([theta_x, phi_g]))
    psi_f = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(f)
    return layers.multiply([x, psi_f])

# 修改 Transformer Block，降低序列长度
def transformer_block(inputs, num_heads, key_dim, ff_dim, rate=0.1):
    # 降低空间维度
    x = layers.AveragePooling2D(pool_size=(2, 2))(inputs)
    height = x.shape[1]
    width = x.shape[2]
    channels = x.shape[3]

    # 展平空间维度
    x = layers.Reshape((height * width, channels))(x)
    x_inp = x

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = layers.Dropout(rate)(x)
    x = layers.Add()([x_inp, x])

    x_inp2 = x
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(ff_dim, activation='relu')(x)
    x = layers.Dropout(rate)(x)
    x = layers.Dense(channels)(x)
    x = layers.Add()([x_inp2, x])

    # 还原空间维度
    x = layers.Reshape((height, width, channels))(x)

    # 恢复原始尺寸
    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)

    return x

# 定义 Hybrid Block，将卷积和Transformer结合
def hybrid_block(x, filters, num_heads=4, key_dim=32, ff_dim=128):
    # 卷积部分
    x_conv = residual_block(x, filters)
    # Transformer部分
    x_trans = transformer_block(x_conv, num_heads=num_heads, key_dim=key_dim, ff_dim=ff_dim)
    # 合并
    x_out = layers.Add()([x_conv, x_trans])
    return x_out

# 定义融合了Transformer的 U-Net 模型
def residual_attention_transformer_unet(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    # Downsampling
    c1 = residual_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = residual_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = hybrid_block(p2, 256, num_heads=4, key_dim=32, ff_dim=128)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = hybrid_block(p3, 512, num_heads=4, key_dim=32, ff_dim=128)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    c5 = hybrid_block(p4, 1024, num_heads=4, key_dim=32, ff_dim=128)

    # Upsampling with Attention
    u4 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(c5)
    att4 = attention_block(c4, u4, 256)
    u4 = layers.concatenate([u4, att4])
    u4 = hybrid_block(u4, 512, num_heads=4, key_dim=32, ff_dim=128)

    u3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(u4)
    att3 = attention_block(c3, u3, 128)
    u3 = layers.concatenate([u3, att3])
    u3 = hybrid_block(u3, 256, num_heads=4, key_dim=32, ff_dim=128)

    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(u3)
    att2 = attention_block(c2, u2, 64)
    u2 = layers.concatenate([u2, att2])
    u2 = residual_block(u2, 128)

    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(u2)
    att1 = attention_block(c1, u1, 32)
    u1 = layers.concatenate([u1, att1])
    u1 = residual_block(u1, 64)

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(u1)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# 初始化数据
batch_size = 8
train_dataset = RoadDataset(os.path.join(data_dir, "Train"), batch_size=batch_size)
val_dataset = RoadDataset(os.path.join(data_dir, "Validation"), batch_size=batch_size)
test_dataset = RoadDataset(os.path.join(data_dir, "Test"), batch_size=batch_size)

# 构建和编译模型
model = residual_attention_transformer_unet()
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=['accuracy'])

# 训练模型
model.fit(train_dataset, validation_data=val_dataset, epochs=50)


import matplotlib.pyplot as plt
# 进行可视化
# 可视化预测结果的函数
def visualize_predictions(model, dataset, indices=None):
    if indices is None:
        raise ValueError("Please provide a list of indices to visualize specific images.")
    
    num_samples = len(indices)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    for idx, img_idx in enumerate(indices):
        # 手动加载指定索引的图片和标签
        image_path = os.path.join(dataset.folder, "image", dataset.image_paths[img_idx])
        label_path = os.path.join(dataset.folder, "label", dataset.label_paths[img_idx])
        
        # 检查图像文件是否有效
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if image is None or label is None:
            print(f"Warning: Skipping invalid image or label file at {image_path} or {label_path}")
            continue  # 跳过无效的图像或标签
        
        image = cv2.resize(image, dataset.img_size) / 255.0
        label = cv2.resize(label, dataset.img_size) / 255.0
        
        # 预测结果
        predictions = model.predict(np.expand_dims(image, axis=0))
        prediction = (predictions[0, :, :, 0] > 0.5).astype(np.uint8)

        # 原始图像
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title("Original Image")
        axes[idx, 0].axis("off")

        # 标签图像
        axes[idx, 1].imshow(label, cmap="gray")
        axes[idx, 1].set_title("Ground Truth")
        axes[idx, 1].axis("off")

        # 预测结果图像
        axes[idx, 2].imshow(prediction, cmap="gray")
        axes[idx, 2].set_title("Prediction")
        axes[idx, 2].axis("off")

    plt.tight_layout()
    plt.show()

# 指定要可视化的6组图片索引
fixed_indices = [0, 1, 2, 3, 4, 5]
visualize_predictions(model, test_dataset, indices=fixed_indices)


# 定义评估指标
def calculate_metrics(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(np.int32)
    y_true = y_true.astype(np.int32)

    iou_metric = MeanIoU(num_classes=2)
    iou_metric.update_state(y_true, y_pred)
    iou = iou_metric.result().numpy()

    dice = np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))

    confusion = confusion_matrix(y_true.flatten(), y_pred.flatten())
    accuracy = np.trace(confusion) / np.sum(confusion)
    precision = confusion[1, 1] / (confusion[1, 1] + confusion[0, 1]) if (confusion[1, 1] + confusion[0, 1]) > 0 else 0
    recall = confusion[1, 1] / (confusion[1, 1] + confusion[1, 0]) if (confusion[1, 1] + confusion[1, 0]) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    oa = np.trace(confusion) / np.sum(confusion)
    pe = np.sum(np.sum(confusion, axis=0) * np.sum(confusion, axis=1)) / (np.sum(confusion) ** 2)
    kappa = (oa - pe) / (1 - pe)

    return {
        "IoU": iou,
        "Dice Coefficient": dice,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
        "Overall Accuracy (OA)": oa,
        "Kappa Coefficient": kappa
    }

# 测试模型并计算评估指标
y_true_list, y_pred_list = [], []
for images, labels in test_dataset:
    predictions = model.predict(images)
    y_true_list.append(labels)
    y_pred_list.append(predictions)

y_true = np.concatenate(y_true_list, axis=0)
y_pred = np.concatenate(y_pred_list, axis=0)

metrics = calculate_metrics(y_true, y_pred)
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")