# -*- coding: utf-8 -*-
import tensorflow as tf
from config import num_class
NUM_CLASS = num_class


class DoubleConv(tf.keras.layers.Layer):
    def __init__(self, filter, kernel):
        super(DoubleConv, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter,
                                            kernel_size=kernel,
                                            strides=1,
                                            padding='same',
                                            activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter,
                                            kernel_size=kernel,
                                            strides=1,
                                            padding='same',
                                            activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        return x


class UP(tf.keras.layers.Layer):
    def __init__(self, filter, kernel):
        super(UP, self).__init__()
        self.upsamp = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.conv1 = tf.keras.layers.Conv2D(filters=filter,
                                            kernel_size=2,
                                            strides=1,
                                            padding='same',
                                            activation='relu')
        self.bn = tf.keras.layers.BatchNormalization()

        self.merge = tf.keras.layers.Concatenate()
        self.doubleconv = DoubleConv(filter=filter, kernel=kernel)

    def call(self, input1, input2, training=False, **kwargs):
        x = self.upsamp(input1)
        x = self.conv1(x)
        x = self.bn(x, training=training)
        x = self.merge([x, input2])
        x = self.doubleconv(x, training=training)

        return x


class UNet(tf.keras.Model):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv64_down = DoubleConv(filter=64, kernel=3)
        self.conv128_down = DoubleConv(filter=128, kernel=3)
        self.conv256_down = DoubleConv(filter=256, kernel=3)
        self.conv512_down = DoubleConv(filter=512, kernel=3)

        self.conv1024 = DoubleConv(filter=1024, kernel=3)

        self.conv512_up = DoubleConv(filter=512, kernel=3)
        self.conv256_up = DoubleConv(filter=256, kernel=3)
        self.conv128_up = DoubleConv(filter=128, kernel=3)

        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

        self.up1 = UP(filter=512, kernel=3)
        self.up2 = UP(filter=256, kernel=3)
        self.up3 = UP(filter=128, kernel=3)
        self.up4 = UP(filter=64, kernel=3)

        self.conv1 = tf.keras.layers.Conv2D(filters=2, kernel_size=3,
                                            activation='relu',
                                            padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=NUM_CLASS, kernel_size=1,
                                            activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        x1 = self.conv64_down(inputs, trainin=training)

        x2 = self.pool1(x1)
        x2 = self.conv128_down(x2, training=training)

        x3 = self.pool2(x2)
        x3 = self.conv256_down(x3, training=training)

        x4 = self.pool3(x3)
        x4 = self.conv512_down(x4, training=training)
        x4 = tf.keras.layers.Dropout(0.5)(x4)

        x5 = self.pool4(x4)
        x5 = self.conv1024(x5, training=training)
        x5 = tf.keras.layers.Dropout(0.5)(x5)

        up1 = self.up1(x5, x4, training=training)
        up2 = self.up2(up1, x3, training=training)
        up3 = self.up3(up2, x2, training=training)
        up4 = self.up4(up3, x1, training=training)

        output = self.conv1(up4)
        output = self.bn(output, training=training)
        output = self.conv2(output)

        return output
