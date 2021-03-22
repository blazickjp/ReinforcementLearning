import tensorflow as tf
import numpy as np

total_size = 10000
seq_len = 10

def make_seq_data(n_samples, seq_len):
    # Boundary tasks
    data, labels = [], []
    for _ in range(n_samples):
        input = np.random.permutation(range(seq_len)).tolist()
        target = sorted(range(len(input)), key=lambda k: input[k])
        data.append(input)
        labels.append(target)
    return np.array(data), np.array(labels)
  
data, labels = make_seq_data(total_size, seq_len)
shifted_labels = np.delete(np.insert(labels, 0, seq_len, axis=1), seq_len, axis=1)

data1, labels1 = make_seq_data(3, 11)
shifted_labels1 = np.insert(labels1, 0, seq_len, axis=1)

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, decoder_state, encoder_states):
    decoder_state = tf.expand_dims(decoder_state, 1)
    score = self.V(tf.nn.tanh(
        self.W1(decoder_state) + self.W2(encoder_states)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * encoder_states
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class Encoder(tf.keras.layers.Layer):
  def __init__(self, vocab_size, units=512):
    super(Encoder, self).__init__()
    self.units = units
    self.vocab_size = vocab_size
    self.lstm = tf.keras.layers.LSTM(self.units, 
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform', input_shape = (None, self.vocab_size))

  def call(self, x):
    x = tf.one_hot(x, depth=self.vocab_size)
    output, state_h, state_c = self.lstm(x)

    return output, state_h, state_c

class Decoder(tf.keras.layers.Layer):
  def __init__(self, vocab_size, units=512):
    super(Decoder, self).__init__()
    self.units = units
    self.vocab_size = vocab_size
    self.attn = BahdanauAttention(units)
    self.lstm = tf.keras.layers.LSTM(self.units,
                                  return_state=True,
                                  return_sequences=True,
                                  recurrent_initializer='glorot_uniform', input_shape = (None, self.vocab_size))

  def call(self, shifted_labels, encoder_output, enc_state_h, enc_state_c):
    x = tf.one_hot(shifted_labels, depth=self.vocab_size)
    out = []
    decoder_output, state_h, state_c = self.lstm(x, initial_state = [enc_state_h, enc_state_c])

    for i in range(decoder_output.shape[1]):
      _, weights = self.attn(decoder_output[:,i,:], encoder_output)
      out.append(tf.squeeze(weights, axis=2))

    return tf.stack(out, axis=2)
  
class PointerNetwork(tf.keras.Model):
    def __init__(self, vocab_size, units):
        super(PointerNetwork, self).__init__()
        self.enc = Encoder(vocab_size, units)
        self.dec = Decoder(vocab_size, units)

    def call(self, input, training=True):
        encoder_output, state_h, state_c = self.enc(input[0])
        if training == False:
            dec_input =  input[1].numpy()
            for i in range(input[1].shape[1] - 1):
                x, scores = self.dec(dec_input, encoder_output, state_h, state_c)
                dec_input[:,i+1] = tf.argmax(scores, axis=2).numpy()[:,i]
            return scores
        else:
            scores = self.dec(input[1], encoder_output, state_h, state_c)

            return tf.transpose(scores, perm=[0,2,1])
          
PntrModel = PointerNetwork(vocab_size=seq_len+1, units=512)
PntrModel.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(), run_eagerly=True, metrics="accuracy")
PntrModel.fit([data, shifted_labels], labels, batch_size=128, epochs=4)


# Unseen data with new input and output size
new_data = np.array([[3,0,1,4], [1,2,3,4,5,6]])
first_input = np.array([[10,1,2,0],[1,2,3]])
out = PntrModel([new_data, first_input])
print(np.argmax(out, axis=2))
