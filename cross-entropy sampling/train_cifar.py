import tensorflow as tf
from tensorflow.keras import Model, datasets, layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# 加载 CIFAR-10 数据集
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 数据预处理
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# 定义 ResNet34 架构
def residual_block(x, filters, strides=(1, 1)):
    identity = x
    x = layers.Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    if strides != (1, 1) or identity.shape[-1] != filters:
        identity = layers.Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(identity)
        identity = layers.BatchNormalization()(identity)
    x = layers.add([x, identity])
    x = layers.Activation('relu')(x)
    return x

def resnet34(input_shape=(32, 32, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = residual_block(x, 128, strides=(2, 2))
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 128)

    x = residual_block(x, 256, strides=(2, 2))
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)

    x = residual_block(x, 512, strides=(2, 2))
    x = residual_block(x, 512)
    x = residual_block(x, 512)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# 创建 ResNet34 模型
model = resnet34()

# 编译模型
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 定义 ModelCheckpoint 回调函数，用于保存最佳模型
checkpoint_callback = ModelCheckpoint('resnet_cifar10.h5', 
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      mode='max',
                                      verbose=1)

# 训练模型
model.fit(train_images, train_labels, 
          epochs=50, 
          batch_size=128, 
          validation_split=0.1, 
          callbacks=[checkpoint_callback])

# 加载最佳模型并评估
best_model = tf.keras.models.load_model('resnet_cifar10.h5')
test_loss, test_acc = best_model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# 提示：最佳模型已保存为 resnet_cifar10.h5
