"""Module for custom layers"""

import tensorflow as tf
import tensorflow.keras.layers as KL

class AnchorsLayer(KL.Layer):
    def __init__(self, anchors, batch_size, **kwargs):
        super(AnchorsLayer, self).__init__(**kwargs)
        self.anchors = tf.Variable(anchors, trainable=False, name='anchors')
        self.batch_size = batch_size

    def call(self, inputs):
        # Duplicate across the batch dimension
        anchors_broadcasted = tf.broadcast_to(self.anchors, (self.batch_size,) + self.anchors.shape)
        return anchors_broadcasted
