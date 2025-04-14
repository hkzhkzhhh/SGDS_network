import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.metrics import MeanIoU
import seaborn as sns

# 数据集路径
data_dir = "Data"

# 自定义数据集加载
class RoadDataset(tf.keras.utils.Sequence):
    def __init__(self, folder, batch_size, img_size=(256, 256), shuffle=True):
        self.image_dir = os.path.join(folder, "image")
        self.label_dir = os.path.join(folder, "label")
        self.image_paths = sorted(os.listdir(self.image_dir))
        self.label_paths = sorted(os.listdir(self.label_dir))
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
            image = cv2.imread(os.path.join(self.image_dir, self.image_paths[i]), cv2.IMREAD_COLOR)
            label = cv2.imread(os.path.join(self.label_dir, self.label_paths[i]), cv2.IMREAD_GRAYSCALE)
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
def residual_block(x, filters, block_id):
    shortcut = layers.Conv2D(filters, (1, 1), padding="same", name=f"residual_shortcut_{block_id}")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", name=f"residual_conv1_{block_id}")(x)
    x = layers.BatchNormalization(name=f"residual_bn1_{block_id}")(x)
    x = layers.ReLU(name=f"residual_relu1_{block_id}")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", name=f"residual_conv2_{block_id}")(x)
    x = layers.BatchNormalization(name=f"residual_bn2_{block_id}")(x)
    x = layers.Add(name=f"residual_add_{block_id}")([shortcut, x])
    x = layers.ReLU(name=f"residual_relu2_{block_id}")(x)
    return x

# 定义 Attention Block
def attention_block(x, g, inter_channels, block_id):
    theta_x = layers.Conv2D(inter_channels, (1, 1), padding="same", name=f"att_theta_{block_id}")(x)
    phi_g = layers.Conv2D(inter_channels, (1, 1), padding="same", name=f"att_phi_{block_id}")(g)
    f = layers.Activation("relu", name=f"att_relu_{block_id}")(layers.Add(name=f"att_add_{block_id}")([theta_x, phi_g]))
    psi_f = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid", name=f"att_psi_{block_id}")(f)
    return layers.Multiply(name=f"att_multiply_{block_id}")([x, psi_f])

# 引入深度可分离卷积和多尺度卷积
def local_stream_advanced(x, filters, block_id):
    # 多尺度卷积：不同尺度的卷积操作并行
    conv1 = layers.Conv2D(filters, (3, 3), padding="same", dilation_rate=(1, 1), name=f"local_conv1_{block_id}")(x)  # 标准卷积
    conv2 = layers.Conv2D(filters, (3, 3), padding="same", dilation_rate=(2, 2), name=f"local_conv2_{block_id}")(x)  # 空洞卷积
    conv3 = layers.Conv2D(filters, (5, 5), padding="same", dilation_rate=(1, 1), name=f"local_conv3_{block_id}")(x)  # 大卷积核

    # 合并多尺度卷积
    x_multi_scale = layers.Concatenate(axis=-1, name=f"local_concat_{block_id}")([conv1, conv2, conv3])

    # 使用深度可分离卷积
    x_separable = layers.SeparableConv2D(filters, (3, 3), padding="same", name=f"local_sepconv_{block_id}")(x_multi_scale)
    x_separable = layers.BatchNormalization(name=f"local_sepbn_{block_id}")(x_separable)
    x_separable = layers.ReLU(name=f"local_seprelu_{block_id}")(x_separable)

    # 残差连接
    shortcut = layers.Conv2D(filters, (1, 1), padding="same", name=f"local_shortcut_{block_id}")(x)
    x = layers.Add(name=f"local_add_{block_id}")([shortcut, x_separable])
    x = layers.ReLU(name=f"local_relu_{block_id}")(x)
    return x

# Swin Transformer Block 的实现
class SwinTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, window_size=8, shift_size=4, name="swin_transformer", **kwargs):
        super(SwinTransformerBlock, self).__init__(name=name, **kwargs)
        assert 0 <= shift_size < window_size, "shift_size must be in 0 <= shift_size < window_size"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm1")
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, name=f"{name}_mha")
        self.drop_path = layers.Dropout(0.1, name=f"{name}_dropout1")  # Drop path rate can be adjusted
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm2")
        self.mlp = models.Sequential([
            layers.Dense(embed_dim * 4, activation='gelu', name=f"{name}_mlp_dense1"),
            layers.Dense(embed_dim, name=f"{name}_mlp_dense2")
        ], name=f"{name}_mlp")
        self.dropout = layers.Dropout(0.1, name=f"{name}_dropout2")

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
    def __init__(self, embed_dim, num_heads=4, dropout_rate=0.1, name="graph_attention", **kwargs):
        super(GraphAttentionBlock, self).__init__(name=name, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, name=f"{name}_mha")
        self.norm = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm")
        self.dropout = layers.Dropout(dropout_rate, name=f"{name}_dropout")
        self.activation = layers.Activation('relu', name=f"{name}_activation")

    def call(self, x):
        # x: (batch, num_nodes, embed_dim)
        x_norm = self.norm(x)
        attn_output = self.attention(x_norm, x_norm)  # (batch, num_nodes, embed_dim)
        attn_output = self.dropout(attn_output)
        x = self.activation(x + attn_output)  # Residual connection and activation
        return x

# 可学习的空间注意力
class LearnableSpatialAttention(layers.Layer):
    def __init__(self, name="learnable_spatial_attention", **kwargs):
        super(LearnableSpatialAttention, self).__init__(name=name, **kwargs)
        self.conv = layers.Conv2D(1, (1, 1), activation="sigmoid", name=f"{name}_conv")

    def call(self, x):
        attention_map = self.conv(x)  # (batch, height, width, 1)
        return layers.Multiply(name=f"{self.name}_multiply")([x, attention_map])

# 自定义 GlobalStreamAdvanced 层
class GlobalStreamAdvanced(layers.Layer):
    def __init__(self, embed_dim=64, num_heads=4, window_size=8, shift_size=4, name="global_stream", **kwargs):
        super(GlobalStreamAdvanced, self).__init__(name=name, **kwargs)
        self.conv = layers.Conv2D(embed_dim, (1, 1), padding="same", name=f"{name}_conv")
        self.swin_block = SwinTransformerBlock(embed_dim, num_heads, window_size=window_size, shift_size=shift_size, name=f"{name}_swin_block")
        self.gnn_block = GraphAttentionBlock(embed_dim, num_heads=num_heads, name=f"{name}_gnn_block")

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
def global_stream(x, embed_dim=64, num_heads=4, window_size=8, shift_size=4, block_id=0):
    global_stream_layer = GlobalStreamAdvanced(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size, name=f"global_stream_{block_id}")
    return global_stream_layer(x)

# 定义融合模块
def fusion_module(local_features, global_features, block_id):
    # 拼接
    fused = layers.Concatenate(axis=-1, name=f"fusion_concat_{block_id}")([local_features, global_features])
    # 可以添加更多融合操作，如卷积、注意力等
    fused = layers.BatchNormalization(name=f"fusion_bn_{block_id}")(fused)
    fused = layers.ReLU(name=f"fusion_relu_{block_id}")(fused)
    return fused

# 定义双流网络的U-Net模型
def dual_stream_unet(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape, name="input_layer")

    # Downsampling部分
    c1 = residual_block(inputs, 64, block_id=1)
    p1 = layers.MaxPooling2D((2, 2), name="pool1")(c1)

    c2 = residual_block(p1, 128, block_id=2)
    p2 = layers.MaxPooling2D((2, 2), name="pool2")(c2)

    # Local and Global Stream 提取特征
    c3_local = local_stream_advanced(p2, 256, block_id=3)  # 使用新的局部特征提取
    c3_global = global_stream(p2, embed_dim=64, num_heads=4, window_size=8, shift_size=4, block_id=3)  # 使用全新的全局特征提取
    c3_fused = fusion_module(c3_local, c3_global, block_id=3)
    p3 = layers.MaxPooling2D((2, 2), name="pool3")(c3_fused)

    c4_local = local_stream_advanced(p3, 512, block_id=4)  # 使用新的局部特征提取
    c4_global = global_stream(p3, embed_dim=64, num_heads=4, window_size=8, shift_size=4, block_id=4)  # 使用全新的全局特征提取
    c4_fused = fusion_module(c4_local, c4_global, block_id=4)
    p4 = layers.MaxPooling2D((2, 2), name="pool4")(c4_fused)

    c5_local = local_stream_advanced(p4, 1024, block_id=5)  # 使用新的局部特征提取
    c5_global = global_stream(p4, embed_dim=64, num_heads=4, window_size=8, shift_size=4, block_id=5)  # 使用全新的全局特征提取
    c5_fused = fusion_module(c5_local, c5_global, block_id=5)

    # Upsampling with Attention
    u4 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same", name="upconv4")(c5_fused)
    u4 = layers.Concatenate(axis=-1, name="concat4")([u4, c4_fused])
    u4 = residual_block(u4, 512, block_id=6)

    u3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same", name="upconv3")(u4)
    u3 = layers.Concatenate(axis=-1, name="concat3")([u3, c3_fused])
    u3 = residual_block(u3, 256, block_id=7)

    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same", name="upconv2")(u3)
    u2 = layers.Concatenate(axis=-1, name="concat2")([u2, c2])
    u2 = residual_block(u2, 128, block_id=8)

    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same", name="upconv1")(u2)
    u1 = layers.Concatenate(axis=-1, name="concat1")([u1, c1])
    u1 = residual_block(u1, 64, block_id=9)

    # 可学习的空间注意力
    u1 = LearnableSpatialAttention(name="learnable_spatial_attention")(u1)

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", name="output_layer")(u1)

    model = models.Model(inputs=[inputs], outputs=[outputs], name="Dual_Stream_U-Net")
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
              metrics=['accuracy', MeanIoU(num_classes=2, name='mean_iou')])

# 打印模型摘要以验证结构
model.summary()

# 定义 Grad-CAM 生成函数
def generate_gradcam(model, image, layer_name):
    """
    生成 Grad-CAM 热图
    :param model: 已训练的模型
    :param image: 输入图像，形状为 (height, width, channels)
    :param layer_name: 要获取梯度的层名称
    :return: Grad-CAM 热图
    """
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(layer_name).output, model.output]
        )
    except ValueError:
        raise ValueError(f"Layer {layer_name} not found in the model.")

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(image, axis=0))
        loss = tf.reduce_mean(predictions)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap = heatmap.numpy()

    return heatmap

# 定义绘制混淆矩阵的函数
def plot_confusion_matrix(y_true, y_pred, class_names=['Background', 'Road']):
    # 确保 y_true 和 y_pred 是二值的
    y_true = (y_true > 0.5).astype(np.int32)
    y_pred = (y_pred > 0.5).astype(np.int32)
    
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=[0,1])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# 定义绘制训练历史的函数
def plot_training_history(history):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # 损失曲线
    axs[0, 0].plot(history.history['loss'], label='Train Loss')
    axs[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axs[0, 0].set_title('Loss Curve')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # 准确率曲线
    axs[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
    axs[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[0, 1].set_title('Accuracy Curve')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()

    # Mean IoU 曲线
    axs[1, 0].plot(history.history['mean_iou'], label='Train Mean IoU')
    axs[1, 0].plot(history.history['val_mean_iou'], label='Validation Mean IoU')
    axs[1, 0].set_title('Mean IoU Curve')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Mean IoU')
    axs[1, 0].legend()

    # 可以在此添加更多的指标曲线
    plt.tight_layout()
    plt.show()

# 特征图可视化
def visualize_feature_maps(model, dataset, image_index=0, layer_names=None):
    if layer_names is None:
        # 默认可视化一些关键层
        layer_names = [
            "residual_conv1_1",
            "local_conv1_3",
            "fusion_relu_3",
            "fusion_concat_3",
            "learnable_spatial_attention_conv"
        ]

    # 创建一个新模型，输出指定的中间层
    intermediate_layers = []
    available_layer_names = [layer.name for layer in model.layers]
    for name in layer_names:
        if name in available_layer_names:
            intermediate_layers.append(model.get_layer(name).output)
        else:
            print(f"Layer {name} not found in the model.")

    if not intermediate_layers:
        raise ValueError("No valid layer names found in the model.")

    intermediate_model = models.Model(inputs=model.input, outputs=intermediate_layers)

    # 获取指定索引的图片和标签
    image_path = os.path.join(dataset.image_dir, dataset.image_paths[image_index])
    label_path = os.path.join(dataset.label_dir, dataset.label_paths[image_index])

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, dataset.img_size) / 255.0

    # 扩展维度以匹配模型输入
    input_image = np.expand_dims(image_resized, axis=0)

    # 获取中间层的输出
    intermediate_outputs = intermediate_model.predict(input_image)

    # 绘制特征图
    for i, feature_map in enumerate(intermediate_outputs):
        num_filters = feature_map.shape[-1]
        if num_filters == 0:
            continue
        # 选择前16个特征图进行可视化
        n_cols = 4
        n_rows = min(4, num_filters // n_cols)
        if n_rows == 0:
            n_rows = 1
        plt.figure(figsize=(n_cols * 3, n_rows * 3))
        plt.suptitle(f"Feature Maps of layer: {layer_names[i]}", fontsize=16)
        for j in range(min(n_rows * n_cols, num_filters)):
            plt.subplot(n_rows, n_cols, j + 1)
            plt.imshow(feature_map[0, :, :, j], cmap='viridis')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

# Grad-CAM 可视化
def visualize_gradcam(model, dataset, image_index=0, layer_name="local_conv3_3"):
    """
    可视化 Grad-CAM 热图
    :param model: 已训练的模型
    :param dataset: 数据集对象
    :param image_index: 要可视化的图片索引
    :param layer_name: 要获取梯度的层名称
    """
    # 获取指定索引的图片和标签
    image_path = os.path.join(dataset.image_dir, dataset.image_paths[image_index])
    label_path = os.path.join(dataset.label_dir, dataset.label_paths[image_index])

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, dataset.img_size) / 255.0

    # 生成 Grad-CAM 热图
    try:
        heatmap = generate_gradcam(model, image_resized, layer_name)
    except Exception as e:
        print(f"Grad-CAM generation failed for image index {image_index}: {e}")
        return

    # 将热图叠加到原始图像上
    heatmap = cv2.resize(heatmap, (dataset.img_size[1], dataset.img_size[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(cv2.cvtColor((image_resized * 255).astype(np.uint8), cv2.COLOR_BGR2RGB), 0.6, heatmap_color, 0.4, 0)

    # 绘制
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor((image_resized * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title("Superimposed Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# 定义训练过程中记录的回调
class TrainingVisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_dataset, val_dataset, num_images=3, gradcam_layer="local_conv3_3"):
        super(TrainingVisualizationCallback, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_images = num_images
        self.gradcam_layer = gradcam_layer
        self.sample_images, self.sample_labels = next(iter(train_dataset))
        self.sample_indices = np.random.choice(len(self.sample_images), self.num_images, replace=False)

    def on_epoch_end(self, epoch, logs=None):
        # 选取样本进行预测
        predictions = self.model.predict(self.sample_images[self.sample_indices])
        predictions = (predictions > 0.5).astype(np.uint8)

        # 绘制样本图像、标签和预测结果
        fig, axes = plt.subplots(self.num_images, 4, figsize=(20, 5 * self.num_images))
        if self.num_images == 1:
            axes = np.expand_dims(axes, axis=0)  # 确保 axes 是二维的
        for i in range(self.num_images):
            # 原始图像
            axes[i, 0].imshow(cv2.cvtColor((self.sample_images[self.sample_indices[i]] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            axes[i, 0].set_title("Original Image")
            axes[i, 0].axis("off")

            # 标签图像
            axes[i, 1].imshow(self.sample_labels[self.sample_indices[i]].squeeze(), cmap="gray")
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis("off")

            # 预测结果图像
            axes[i, 2].imshow(predictions[i].squeeze(), cmap="gray")
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis("off")

            # Grad-CAM 热图
            try:
                heatmap = generate_gradcam(self.model, self.sample_images[self.sample_indices[i]], layer_name=self.gradcam_layer)
                heatmap = cv2.resize(heatmap, (self.train_dataset.img_size[1], self.train_dataset.img_size[0]))
                heatmap = np.uint8(255 * heatmap)
                heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = cv2.addWeighted(cv2.cvtColor((self.sample_images[self.sample_indices[i]] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB), 0.6, heatmap_color, 0.4, 0)
                axes[i, 3].imshow(superimposed_img)
                axes[i, 3].set_title("Grad-CAM")
                axes[i, 3].axis("off")
            except Exception as e:
                print(f"Grad-CAM generation failed for sample {self.sample_indices[i]}: {e}")
                axes[i, 3].axis("off")

        plt.tight_layout()
        plt.show()

# 进行可视化预测
def visualize_predictions(model, dataset, indices=None, gradcam_layer="local_conv3_3"):
    if indices is None:
        raise ValueError("Please provide a list of indices to visualize specific images.")
    
    num_samples = len(indices)
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)  # 确保 axes 是二维的
    
    for idx, img_idx in enumerate(indices):
        # 手动加载指定索引的图片和标签
        image_path = os.path.join(dataset.image_dir, dataset.image_paths[img_idx])
        label_path = os.path.join(dataset.label_dir, dataset.label_paths[img_idx])
        
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dataset.img_size) / 255.0
        label = cv2.resize(label, dataset.img_size) / 255.0
        
        # 预测结果
        predictions = model.predict(np.expand_dims(image, axis=0))
        prediction = (predictions[0, :, :, 0] > 0.5).astype(np.uint8)

        # 生成 Grad-CAM 热图
        try:
            heatmap = generate_gradcam(model, image, gradcam_layer)
            heatmap = cv2.resize(heatmap, (dataset.img_size[1], dataset.img_size[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB), 0.6, heatmap_color, 0.4, 0)
        except Exception as e:
            print(f"Grad-CAM generation failed for image index {img_idx}: {e}")
            superimposed_img = np.zeros_like(image)
        
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

        # Grad-CAM 热图
        axes[idx, 3].imshow(superimposed_img)
        axes[idx, 3].set_title("Grad-CAM")
        axes[idx, 3].axis("off")
    
    plt.tight_layout()
    plt.show()

# 训练模型并记录训练历史
history = model.fit(
    train_dataset, 
    validation_data=val_dataset, 
    epochs=50, 
    callbacks=[
        TrainingVisualizationCallback(
            train_dataset, 
            val_dataset, 
            num_images=3, 
            gradcam_layer="local_conv3_3"
        )
    ]
)

# 绘制训练和验证的损失与准确率曲线
plot_training_history(history)

# 测试模型并计算评估指标
def calculate_metrics(y_true, y_pred):
    # 将 y_true 和 y_pred 二值化
    y_pred = (y_pred > 0.5).astype(np.int32)
    y_true = (y_true > 0.5).astype(np.int32)  # 修正：对 y_true 进行二值化

    iou_metric = MeanIoU(num_classes=2)
    iou_metric.update_state(y_true, y_pred)
    iou = iou_metric.result().numpy()

    dice = (2 * np.sum(y_pred[y_true == 1]) + 1e-7) / (np.sum(y_pred) + np.sum(y_true) + 1e-7)

    confusion = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=[0,1])
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

# 绘制混淆矩阵
plot_confusion_matrix(y_true, y_pred, class_names=['Background', 'Road'])

# 指定要可视化的6组图片索引
fixed_indices = [0, 1, 2, 3, 4, 5]
visualize_predictions(model, test_dataset, indices=fixed_indices, gradcam_layer="local_conv3_3")

# Grad-CAM 可视化示例
visualize_gradcam(model, test_dataset, image_index=0, layer_name="local_conv3_3")
