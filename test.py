# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import cv2
from Unet import UNet
from dataset import DataGenerator


if not os.path.exists("./results"): os.mkdir("./results")

testSet = DataGenerator("data/test", batch_size=1)

model = UNet()
model.build(input_shape=(None, 256, 256, 1))
model.load_weights("model.h5")
print('loaded weights file successful')


'''
# Get the accuracy on the test set
loss_object = tf.keras.metrics.SparseCategoricalCrossentropy()
test_loss = tf.keras.metrics.Mean()
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


def test_step(images, labels):
    pred = model(images, training=False)
    t_loss = loss_object(labels, pred)
    test_loss(t_loss)
    test_accuracy(labels, pred)


for img, labels in testSet:
    test_step(img, labels)
    print("loss: {:.5f}, test accuracy: {:.5f}".format(test_loss.result(),
                                                       test_accuracy.result()))

print("The accuracy on test set is: {:.3f}%".format(test_accuracy.result() * 100))

'''

alpha = 0.3
for idx, (img, mask) in enumerate(testSet):
    oring_img = img[0]
    pred_mask = model.predict(img)[0]
    pred_mask[pred_mask > 0.5] = 1
    pred_mask[pred_mask <= 0.5] = 0
    img = cv2.cvtColor(img[0], cv2.COLOR_GRAY2RGB)
    H, W, C = img.shape
    for i in range(H):
        for j in range(W):
            if pred_mask[i][j][0] <= 0.5:
                img[i][j] = (1-alpha)*img[i][j]*255 + alpha*np.array([0, 0, 255])
            else:
                img[i][j] = img[i][j]*255
    image_accuracy = np.mean(mask == pred_mask)
    image_path = "./results/pred_"+str(idx)+".png"
    print("=> accuracy: %.4f, saving %s" % (image_accuracy, image_path))
    cv2.imwrite(image_path, img)
    cv2.imwrite("./results/origin_%d.png" % idx, oring_img*255)
    if idx == 29: break
