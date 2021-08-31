import tensorflow as tf
import matplotlib.pyplot as plt

class RDB:

  def __init__(self, conv_number, filter, input_shape):
    input = tf.keras.layers.Input(shape=input_shape)
    last_output = [input]
    outputs = []
    for i in range(conv_number):
      outputs.append(tf.keras.layers.Conv2D(filter, (3, 3), activation='relu', padding='same')(last_output[-1]))
      last_output.append(tf.keras.layers.Concatenate()([last_output[-1], outputs[-1]]))
    last_conv = tf.keras.layers.Conv2D(input_shape[2], (3, 3), padding='same')(last_output[-1])
    add_1 = tf.keras.layers.Add()([last_conv, input])
    final_output = tf.keras.layers.Activation('relu')(add_1)
    self.model = tf.keras.Model(input, final_output)
  
  def summary(self):
    self.model.summary()
  
  def get_model(self):
    return self.model
  
class RRDB:

  def __init__(self, blocks_number, conv_number, filter, input_shape):
    input = tf.keras.layers.Input(shape=input_shape)
    last_output = [input]
    outputs = []
    for i in range(blocks_number):
      rdb = RDB(conv_number, filter, input_shape)
      rdb_model = rdb.get_model()
      outputs.append(rdb_model(last_output[-1]))
      last_output.append(tf.keras.layers.Add()([last_output[-1], outputs[-1]]))
    final_output = tf.keras.layers.Add()([input, last_output[-1]])
    self.model = tf.keras.Model(input, final_output)
  
  def summary(self):
    self.model.summary()
  
  def get_model(self):
    return self.model
  
class RRDN:

  def __init__(self, rdb_blocks_number, conv_rdb_number, rdb_filter, input_shape, task='LR->HR'):
    self.task = task
    if task == 'LR->HR':
      input = tf.keras.layers.Input(shape=input_shape)
      conv_1 = tf.keras.layers.Conv2D(rdb_filter, (3, 3), activation='relu', padding='same')(input)
      rrdb = RRDB(rdb_blocks_number, conv_rdb_number, rdb_filter, (input_shape[0], input_shape[1], rdb_filter)).get_model()(conv_1)
      conv_2 = tf.keras.layers.Conv2D(rdb_filter, (3, 3), activation='relu', padding='same')(rrdb)
      add_1 = tf.keras.layers.Add()([conv_1, conv_2])
      upsampling_1 = tf.keras.layers.UpSampling2D((2,2))(add_1)
      conv_3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(upsampling_1)
      up_sampling_2 = tf.keras.layers.UpSampling2D((2,2))(conv_3)
      conv_4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up_sampling_2) 
      output = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv_4) 
      self.model = tf.keras.Model(input, output)
    elif task == 'HR->LR':
      input = tf.keras.layers.Input(shape=input_shape)
      conv_1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input)
      maxpooling_1 = tf.keras.layers.MaxPooling2D((2,2))(conv_1)
      conv_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(maxpooling_1)
      maxpooling_2 = tf.keras.layers.MaxPooling2D((2,2))(conv_2)
      conv_3 = tf.keras.layers.Conv2D(rdb_filter, (3, 3), activation='relu', padding='same')(maxpooling_2)
      rrdb = RRDB(rdb_blocks_number, conv_rdb_number, rdb_filter, (input_shape[0]//4, input_shape[1]//4, rdb_filter)).get_model()(conv_3)
      add_1 = tf.keras.layers.Add()([maxpooling_2, rrdb])
      conv_4 = tf.keras.layers.Conv2D(rdb_filter, (3, 3), activation='relu', padding='same')(add_1)
      output = tf.keras.layers.Conv2D(3, (3, 3), activation='relu', padding='same')(conv_4)
      self.model = tf.keras.Model(input, output)
    
  def summary(self):
    self.model.summary()
  
  def train(self, x_train, y_train, x_val, y_val, optimizer='adamax', batch_size=8, loss='mse', epochs=50, display=False, save_path=''):
    self.model.compile(optimizer=optimizer,loss=loss)
    if len(save_path)==0:
      print('Please enter the path to save models')
    else:
      checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='{}{}.hdf5'.format(save_path,self.task), monitor='val_loss', save_best_only=True, mode='auto')
      callbacks_list = [checkpoint]
      history=self.model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), callbacks=callbacks_list) 
    if display:
      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'validation'], loc='upper left')
      plt.show()
    self.model.save('{}{}.hdf5'.format(save_path, self.task))

    def eval(self, X, Y):
      return self.model.evaluate(X, Y, batch_size=8)
    
    def pred(self, X):
      return self.model.predict(X)
    
    def load_model(path):
      self.model.load_weights(path)