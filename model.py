import tensorflow as tf
from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Input, Lambda, Dropout
from keras.layers.merging import concatenate


def unet(IMG_WIDTH=768, IMG_HEIGHT=768, IMG_CHANNELS=3):
    # определение входных данных
    inputs = Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    # нормализация
    s = Lambda(lambda x: x / 255)(inputs)

    # кол-во фильтров небольшое, так как не хватает памяти
    conv1 = Conv2D(4, (3, 3), activation='relu', padding='same') (s)
    conv1 = Dropout(0.1) (conv1)
    conv1 = Conv2D(4, (3, 3), activation='relu', padding='same') (conv1)
    max_pool1 = MaxPooling2D((2, 2)) (conv1)

    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same') (max_pool1)
    conv2 = Dropout(0.1) (conv2)
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same') (conv2)
    max_pool2 = MaxPooling2D((2, 2)) (conv2)

    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same') (max_pool2)
    conv3 = Dropout(0.2) (conv3)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same') (conv3)
    max_pool3 = MaxPooling2D((2, 2)) (conv3)

    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same') (max_pool3)
    conv4 = Dropout(0.2) (conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same') (conv4)
    max_pool4 = MaxPooling2D(pool_size=(2, 2)) (conv4)

    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same') (max_pool4)
    conv5 = Dropout(0.3) (conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same') (conv5)

    transpose6 = Conv2DTranspose(64, (2, 2), strides=(2,2), padding='same') (conv5)
    transpose6 = concatenate([transpose6, conv4])
    conv6 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same') (transpose6)
    conv6 = Dropout(0.2) (conv6)
    conv6 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same') (conv6)

    transpose7 = Conv2DTranspose(32, (2, 2), strides=(2,2), padding='same') (conv6)
    transpose7 = concatenate([transpose7, conv3])
    conv7 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same') (transpose7)
    conv7 = Dropout(0.2) (conv7)
    conv7 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same') (conv7)

    transpose8 = Conv2DTranspose(16, (2, 2), strides=(2,2), padding='same') (conv7)
    transpose8 = concatenate([transpose8, conv2])
    conv8 = Conv2D(16, (3, 3), strides=1, activation='relu', padding='same') (transpose8)
    conv8 = Dropout(0.1) (conv8)
    conv8 = Conv2D(16, (3, 3), strides=1, activation='relu', padding='same') (conv8)

    transpose9 = Conv2DTranspose(4, (2, 2), strides=(2,2), padding='same') (conv8)
    transpose9 = concatenate([transpose9, conv1], axis=3)
    conv9 = Conv2D(4, (3, 3), strides=1, activation='relu', padding='same') (transpose9)
    conv9 = Dropout(0.1) (conv9)
    conv9 = Conv2D(4, (3, 3), strides=1, activation='relu', padding='same') (conv9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv9)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
    return model
