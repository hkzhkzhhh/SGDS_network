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
        self.image_dir = os.path.join(folder, "image")
        self.label_dir = os.path.join(folder, "label")
        self.image_paths = sorted([f for f in os.listdir(self.image_dir) if not f.startswith('.')])  # 排除隐藏文件
        self.label_paths = sorted([f for f in os.listdir(self.label_dir) if not f.startswith('.')])  # 排除隐藏文件
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, labels = [], []
        for i in batch_indices:
            image_path = os.path.join(self.image_dir, self.image_paths[i])
            label_path = os.path.join(self.label_dir, self.label_paths[i])

            # 加载图像并检查是否为空
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                raise ValueError(f"Failed to load image at {image_path}")
            if label is None:
                raise ValueError(f"Failed to load label at {label_path}")

            image = cv2.resize(image, self.img_size)
            label = cv2.resize(label, self.img_size)
            images.append(image / 255.0)
            labels.append(label / 255.0)
        return np.array(images, dtype=np.float32), np.expand_dims(np.array(labels, dtype=np.float32), axis=-1)

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
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    return x


# 引入深度可分离卷积和多尺度卷积
def local_stream_advanced(x, filters):
    # 多尺度卷积：不同尺度的卷积操作并行
    conv1 = layers.Conv2D(filters, (3, 3), padding="same", dilation_rate=(1, 1))(x)  # 标准卷积
    conv2 = layers.Conv2D(filters, (3, 3), padding="same", dilation_rate=(2, 2))(x)  # 空洞卷积
    conv3 = layers.Conv2D(filters, (5, 5), padding="same", dilation_rate=(1, 1))(x)  # 大卷积核

    # 合并多尺度卷积
    x_multi_scale = layers.Concatenate(axis=-1)([conv1, conv2, conv3])

    # 使用深度可分离卷积
    x_separable = layers.SeparableConv2D(filters, (3, 3), padding="same")(x_multi_scale)
    x_separable = layers.BatchNormalization()(x_separable)
    x_separable = layers.ReLU()(x_separable)

    # 残差连接
    shortcut = layers.Conv2D(filters, (1, 1), padding="same")(x)
    x = layers.Add()([shortcut, x_separable])
    x = layers.ReLU()(x)
    return x

# Swin Transformer Block 的实现
class SwinTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, window_size=8, shift_size=4, **kwargs):
        super(SwinTransformerBlock, self).__init__(**kwargs)
        assert 0 <= shift_size < window_size, "shift_size must be in 0 <= shift_size < window_size"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.drop_path = layers.Dropout(0.1)  # Drop path rate can be adjusted
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = models.Sequential([
            layers.Dense(embed_dim * 4, activation='gelu'),
            layers.Dense(embed_dim)
        ])
        self.dropout = layers.Dropout(0.1)

    def window_partition(self, x):
        # x: (batch, height, width, channels)
        batch, height, width, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, (batch, height // self.window_size, self.window_size, width // self.window_size, self.window_size, channels))
        windows = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])  # (batch, num_windows_h, num_windows_w, window_size, window_size, channels)
        windows = tf.reshape(windows, (-1, self.window_size * self.window_size, channels))  # (batch*num_windows, window_size*window_size, channels)
        return windows

    def window_reverse(self, windows, height, width):
        # windows: (batch*num_windows, window_size*window_size, channels)
        batch = tf.shape(windows)[0] // (height // self.window_size * width // self.window_size)
        x = tf.reshape(windows, (batch, height // self.window_size, width // self.window_size, self.window_size, self.window_size, self.embed_dim))
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])  # (batch, num_windows_h, window_size, num_windows_w, window_size, channels)
        x = tf.reshape(x, (batch, height, width, self.embed_dim))
        return x

    def call(self, x):
        batch, height, width, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        # Shifted Window
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        windows = self.window_partition(shifted_x)  # (batch*num_windows, window_size*window_size, channels)

        # Multi-Head Self-Attention
        windows_norm = self.norm1(windows)
        attn_output = self.attn(windows_norm, windows_norm)  # (batch*num_windows, window_size*window_size, channels)
        attn_output = self.drop_path(attn_output)
        windows = windows + attn_output  # Residual connection

        # MLP
        windows_norm = self.norm2(windows)
        mlp_output = self.mlp(windows_norm)
        mlp_output = self.dropout(mlp_output)
        windows = windows + mlp_output  # Residual connection

        # Reverse windows
        x = self.window_reverse(windows, height, width)  # (batch, height, width, channels)

        # Reverse shift
        if self.shift_size > 0:
            x = tf.roll(x, shift=(self.shift_size, self.shift_size), axis=(1, 2))

        return x

# 图神经网络（GNN）模块的实现（Graph Attention Network）
class GraphAttentionBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads=4, dropout_rate=0.1, **kwargs):
        super(GraphAttentionBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)
        self.activation = layers.Activation('relu')

    def call(self, x):
        # x: (batch, num_nodes, embed_dim)
        x_norm = self.norm(x)
        attn_output = self.attention(x_norm, x_norm)  # (batch, num_nodes, embed_dim)
        attn_output = self.dropout(attn_output)
        x = self.activation(x + attn_output)  # Residual connection and activation
        return x

# 可学习的空间注意力
class LearnableSpatialAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(LearnableSpatialAttention, self).__init__(**kwargs)
        self.conv = layers.Conv2D(1, (1, 1), activation="sigmoid")

    def call(self, x):
        attention_map = self.conv(x)  # (batch, height, width, 1)
        return layers.Multiply()([x, attention_map])

# 自定义 GlobalStreamAdvanced 层
class GlobalStreamAdvanced(layers.Layer):
    def __init__(self, embed_dim=64, num_heads=4, window_size=8, shift_size=4, **kwargs):
        super(GlobalStreamAdvanced, self).__init__(**kwargs)
        self.conv = layers.Conv2D(embed_dim, (1, 1), padding="same")
        self.swin_block = SwinTransformerBlock(embed_dim, num_heads, window_size=window_size, shift_size=shift_size)
        self.gnn_block = GraphAttentionBlock(embed_dim, num_heads=num_heads)

    def call(self, x):
        x = self.conv(x)  # (batch, height, width, embed_dim)
        x = self.swin_block(x)  # (batch, height, width, embed_dim)

        # Reshape to graph format: (batch, num_nodes, embed_dim)
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        embed_dim = tf.shape(x)[3]
        num_nodes = height * width
        x_graph = tf.reshape(x, (batch_size, num_nodes, embed_dim))  # (batch, num_nodes, embed_dim)

        # GNN Module
        x_graph = self.gnn_block(x_graph)  # (batch, num_nodes, embed_dim)

        # Reshape back to feature map: (batch, height, width, embed_dim)
        x = tf.reshape(x_graph, (batch_size, height, width, embed_dim))
        return x

# Global Stream 架构
def global_stream(x, embed_dim=64, num_heads=4, window_size=8, shift_size=4):
    global_stream_layer = GlobalStreamAdvanced(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size)
    return global_stream_layer(x)

# 定义融合模块
def fusion_module(local_features, global_features):
    # 拼接
    fused = layers.Concatenate(axis=-1)([local_features, global_features])
    return fused

# 定义双流网络的U-Net模型
def dual_stream_unet(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    # Downsampling部分
    c1 = residual_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = residual_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Local and Global Stream 提取特征
    c3_local = local_stream_advanced(p2, 256)  # 使用新的局部特征提取
    c3_global = global_stream(p2, embed_dim=64, num_heads=4, window_size=8, shift_size=4)  # 使用全新的全局特征提取
    c3_fused = fusion_module(c3_local, c3_global)
    p3 = layers.MaxPooling2D((2, 2))(c3_fused)

    c4_local = local_stream_advanced(p3, 512)  # 使用新的局部特征提取
    c4_global = global_stream(p3, embed_dim=64, num_heads=4, window_size=8, shift_size=4)  # 使用全新的全局特征提取
    c4_fused = fusion_module(c4_local, c4_global)
    p4 = layers.MaxPooling2D((2, 2))(c4_fused)

    c5_local = local_stream_advanced(p4, 1024)  # 使用新的局部特征提取
    c5_global = global_stream(p4, embed_dim=64, num_heads=4, window_size=8, shift_size=4)  # 使用全新的全局特征提取
    c5_fused = fusion_module(c5_local, c5_global)

    # Upsampling with Attention
    u4 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(c5_fused)
    u4 = layers.Concatenate(axis=-1)([u4, c4_fused])
    u4 = residual_block(u4, 512)

    u3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(u4)
    u3 = layers.Concatenate(axis=-1)([u3, c3_fused])
    u3 = residual_block(u3, 256)

    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(u3)
    u2 = layers.Concatenate(axis=-1)([u2, c2])
    u2 = residual_block(u2, 128)

    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(u2)
    u1 = layers.Concatenate(axis=-1)([u1, c1])
    u1 = residual_block(u1, 64)

    # 可学习的空间注意力
    u1 = LearnableSpatialAttention()(u1)

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(u1)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# 初始化数据
batch_size = 8
train_dataset = RoadDataset(os.path.join(data_dir, "Train"), batch_size=batch_size)
val_dataset = RoadDataset(os.path.join(data_dir, "Validation"), batch_size=batch_size)
test_dataset = RoadDataset(os.path.join(data_dir, "Test"), batch_size=batch_size)

# 构建和编译模型
model = dual_stream_unet()
model.compile(optimizer='adam', 
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), 
              metrics=['accuracy'])

# 打印模型摘要以验证结构
model.summary()

# 训练模型
model.fit(train_dataset, validation_data=val_dataset, epochs=50)

# 可视化函数
def visualize_predictions(model, dataset, indices=None):
    if indices is None:
        raise ValueError("Please provide a list of indices to visualize specific images.")
    
    num_samples = len(indices)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    for idx, img_idx in enumerate(indices):
        # 手动加载指定索引的图片和标签
        image_path = os.path.join(dataset.image_dir, dataset.image_paths[img_idx])
        label_path = os.path.join(dataset.label_dir, dataset.label_paths[img_idx])
        
        # 加载图像和标签并检查是否为空
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")
        if label is None:
            raise ValueError(f"Failed to load label at {label_path}")
        
        image = cv2.resize(image, dataset.img_size) / 255.0
        label = cv2.resize(label, dataset.img_size) / 255.0
        
        # 预测结果
        predictions = model.predict(np.expand_dims(image, axis=0))
        prediction = (predictions[0, :, :, 0] > 0.5).astype(np.uint8)

        # 原始图像
        axes[idx, 0].imshow(cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
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

# 调用可视化函数
fixed_indices = [0, 1, 2, 3, 4, 5]
visualize_predictions(model, test_dataset, indices=fixed_indices)
   

# 定义评估指标
def calculate_metrics(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(np.int32)
    y_true = y_true.astype(np.int32)

    iou_metric = MeanIoU(num_classes=2)
    iou_metric.update_state(y_true, y_pred)
    iou = iou_metric.result().numpy()

    dice = (2 * np.sum(y_pred[y_true == 1]) + 1e-7) / (np.sum(y_pred) + np.sum(y_true) + 1e-7)

    confusion = confusion_matrix(y_true.flatten(), y_pred.flatten())
    accuracy = np.trace(confusion) / np.sum(confusion)
    precision = confusion[1, 1] / (confusion[1, 1] + confusion[0, 1] + 1e-7)
    recall = confusion[1, 1] / (confusion[1, 1] + confusion[1, 0] + 1e-7)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    oa = accuracy  # Overall Accuracy
    pe = np.sum(np.sum(confusion, axis=0) * np.sum(confusion, axis=1)) / (np.sum(confusion) ** 2 + 1e-7)
    kappa = (oa - pe) / (1 - pe + 1e-7)

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
