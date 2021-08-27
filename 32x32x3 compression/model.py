import tensorflow as tf 
import matplotlib.pyplot as plt

class Model:

  def __init__(self, input_shape): 
    encoder_input=tf.keras.layers.Input(shape=input_shape)

    x=tf.keras.layers.Conv2D(256, (3,3), activation='sigmoid', padding='same')(encoder_input)
    x=tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x=tf.keras.layers.MaxPooling2D((2,2), padding='same')(x)
    x=tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x=tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x=tf.keras.layers.MaxPooling2D((2,2), padding='same')(x)
    x=tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x=tf.keras.layers.MaxPooling2D((2,2), padding='same')(x)
    encoder_output=tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)

    self.encoder=tf.keras.Model(encoder_input, encoder_output)

    decoder_input=tf.keras.layers.Input(shape=self.encoder.layers[-1].output_shape[1:])
    x=tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(decoder_input)
    x=tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x=tf.keras.layers.UpSampling2D((2,2))(x)
    x=tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x=tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x=tf.keras.layers.UpSampling2D((2,2))(x)
    x=tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x=tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x=tf.keras.layers.UpSampling2D((2,2))(x)
    decoder_output=tf.keras.layers.Conv2D(input_shape[2], (3,3), activation='sigmoid', padding='same')(x)

    self.decoder=tf.keras.Model(decoder_input, decoder_output)

    encoded=self.encoder(encoder_input)
    rebuild=self.decoder(encoded)
    
    self.autoencoder=tf.keras.Model(encoder_input, rebuild)
  
  def summary(self):
    print('encoder : \n')
    self.encoder.summary()
    print('\ndecoder : \n')
    self.decoder.summary()
  
  def train(self, X, optimizer='adamax', batch_size=32, loss='mse', val_split=0.15, epochs=50, display=False, save_path=''):
    X_train=X/255.
    self.autoencoder.compile(optimizer=optimizer,loss=loss)
    if len(save_path)==0:
      print('Please enter the path to save models')
    else:
      checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_path+'autoencoder.hdf5', monitor='val_loss', save_best_only=True, mode='auto')
      callbacks_list = [checkpoint]
      history=self.autoencoder.fit(x=X_train, y=X_train, batch_size=batch_size, epochs=epochs, validation_split=val_split, callbacks=callbacks_list) 
    if display:
      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'validation'], loc='upper left')
      plt.show()
    self.encoder.save(save_path+'encoder.hdf5')
    self.decoder.save(save_path+'decoder.hdf5')

  def load_autoencoder(self, path):
    self.autoencoder.compile('adam', loss='mse')
    self.autoencoder.load_weights(path)
  
  def load_encoder(self, path):
    self.encoder.load_weights(path)

  def load_decoder(self, path):
    self.decoder.load_weights(path)
  
  def eval(self, X):
    return self.autoencoder.evaluate(X/255., X/255., batch_size=32)
    
  def encode(self, X):
    return self.encoder.predict(X/255.)
  
  def decode(self, X):
    return (self.decoder.predict(X)*255.).round().astype(int)
