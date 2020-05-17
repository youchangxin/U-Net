# -*- coding: utf-8 -*-
import tensorflow as tf
import os

from Unet import UNet
from dataset import DataGenerator
from tensorflow.keras.optimizers import Adam
from config import lr, batch_size


model = UNet()
trainset = DataGenerator("data/train", batch_size=batch_size)

model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])
model.fit_generator(trainset, steps_per_epoch=1250, epochs=5, verbose=1)
model.save_weights("model.h5")
