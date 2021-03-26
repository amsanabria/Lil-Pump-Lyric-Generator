import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import sys
import os 
import time

# Read the input
text = open('lyrics.txt', 'rb').read().decode(encoding='utf-8')

# Process the text
vocab = sorted(set(text))

ids_from_chars = preprocessing.StringLookup(
    vocabulary=list(vocab))

chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True)

def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))

ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

seq_length = 100
examples_per_epoch = len(text)

sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# Create trainting batches
BATCH_SIZE = 64

BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))


vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024



# Build first model
class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True, 
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else: 
      return x
  
model = MyModel(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)

# Build OneStep model
class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature=temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    skip_ids = self.ids_from_chars(['','[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(

        values=[-float('inf')]*len(skip_ids),
        indices = skip_ids,

        dense_shape=[len(ids_from_chars.get_vocabulary())]) 
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):

    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    predicted_logits, states =  self.model(inputs=input_ids, states=states, 
                                          return_state=True)
    
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature

    predicted_logits = predicted_logits + self.prediction_mask


    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    predicted_chars = self.chars_from_ids(predicted_ids)

    return predicted_chars, states

# Is this loss?
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

# Print Menu
while True:
    os.system('cls')
    flag = 1
    
    print("\n" + "-"*5 + "Lil Pump Lyric Generator" + "-"*5 + "\n")
    print("1. Generate Lyrics")
    print("2. Train Model")
    print("3. Load Custom Model")
    print("4. Load Pretrained Model")
    print("\n" + "."*15 + "Using TF " + tf.__version__)
    
    select = input()
    
    if select.isdigit() and select == '1':
        print("\nEnter first word or phrase:")
        first = input()
        
        one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
        
        start = time.time()
        states = None
        next_char = tf.constant([first])
        result = [next_char]
        
        for n in range(1000):
          next_char, states = one_step_model.generate_one_step(next_char, states=states)
          result.append(next_char)
        
        result = tf.strings.join(result)
        end = time.time()
        
        print("_"*80)
        
        print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
        
        print(f"\nRun time: {end - start}")
    
    elif select.isdigit() and select == '2':
        while(flag == 1):
            print("\nFor how many epochs?")
            EPOCHS = input()
            if EPOCHS.isdigit():
                flag = 0;
                EPOCHS = int(EPOCHS)
                
                model.compile(optimizer='adam', loss=loss)
                
                # Train model
                model.fit(dataset, epochs=EPOCHS)
                
                print("Save model? Y/n")
                save = input()
                
                if save.lower() == 'y' or save.lower() == 'yes':
                    # Save model
                    model.save_weights('one_step')
                    print("MODEL SAVED")
        
    elif select.isdigit() and int(select) == 3:
        model.load_weights('one_step')
        model.compile(optimizer='adam', loss=loss)
        print("MODEL LOADED!!!")

    elif select.isdigit() and int(select) == 3:
        model.load_weights('pretrained')
        model.compile(optimizer='adam', loss=loss)
        print("MODEL LOADED!!!")
    
    print("."*80)
    print("Press ENTER to continue")
    x = input()
    os.system('cls')