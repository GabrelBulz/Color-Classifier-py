import tensorflow as tf 

import json

labelList = [
  'red-ish',
  'green-ish',
  'blue-ish',
  'orange-ish',
  'yellow-ish',
  'pink-ish',
  'purple-ish',
  'brown-ish',
  'grey-ish'
]


filename = 'C:\\Users\\bulzg\\Desktop\\colorData.json'

with open(filename) as json_file:
    entries = json.load(json_file)

data = entries['entries']

colors = []
labels = []
for i in data:
    col = [i['r']/255, i['g']/255, i['b']/255]
    colors.append(col)
    labels.append(labelList.index(i['label']))


input = tf.constant(colors)
outputs = tf.one_hot(labels,9)

print(outputs[1:3])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)))
model.add(tf.keras.layers.Dense(9, activation = 'softmax'))

learning_rate = 0.2
optimizer = tf.keras.optimizers.SGD(learning_rate)

model.compile(optimizer=optimizer, loss = tf.keras.losses.CategoricalCrossentropy(), validation_split = 0.1)

model.fit(input, outputs, epochs=300, verbose=1)

model.save('trained_model.h5')
