import tensorflow as tf

class OneHotEncodingLayer(tf.keras.layers.experimental.preprocessing.PreprocessingLayer):
  def __init__(self, vocabulary=None, depth=None, minimum=None):
    super().__init__()
    self.vectorization = tf.keras.layers.TextVectorization(output_sequence_length=1)

    if vocabulary:
      self.vectorization.set_vocabulary(vocabulary)
    self.depth = depth
    self.minimum = minimum

  def adapt(self, data):
    self.vectorization.adapt(data)
    vocab = self.vectorization.get_vocabulary()
    self.depth = len(vocab)
    indices = [i[0] for i in self.vectorization([[v] for v in vocab]).numpy()]
    self.minimum = min(indices)

  def call(self,inputs):
    vectorized = self.vectorization.call(inputs)
    subtracted = tf.subtract(vectorized, tf.constant([self.minimum], dtype=tf.int64))
    encoded = tf.one_hot(subtracted, self.depth)
    return tf.keras.layers.Reshape((self.depth,))(encoded)

  def get_config(self):
    return {'vocabulary': self.vectorization.get_vocabulary(), 'depth': self.depth, 'minimum': self.minimum}
