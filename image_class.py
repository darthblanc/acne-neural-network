import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

import keras


img_height = 640
img_width = 640

Sequential = keras.models.Sequential
layers = keras.layers

model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# # model.summary()
# batch_size = 32

load_model = keras.models.load_model
model = load_model('checkpoint.h5')


train_ds = keras.utils.image_dataset_from_directory(
  r'data-2/train',
  image_size=(img_height, img_width),
  # batch_size=batch_size
  )
class_names = train_ds.class_names

val_ds = keras.preprocessing.image_dataset_from_directory(
  r'data-2/valid',
  image_size=(img_height, img_width),
#   batch_size=batch_size
  )

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

model.save('checkpoint.h5')

image_path = 'data-2/valid/acne/acne-90_jpeg.rf.715856ba33d2f7c26294ceea4aaa611b.jpg'
image  = keras.utils.load_img(image_path, target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(image)
img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
