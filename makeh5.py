from tensorflow import keras
model = keras.models.load_model('slr_model')

model.save('slr_model.h5')