import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.utils.vis_utils import plot_model
import matplotlib.style as style
from keras.models import load_model


cat_columns = ['job','marital','education','default','housing','loan','contact','month', 'previous','poutcome']

continous = ['age','balance','day','campaign','pdays','previous']

label = ['y']

ds = pd.read_csv('outbound_campaign_dataset.csv',sep=';')
# discarding duration column from ds (refer to readme for justification)
ds = ds.drop(['duration'],axis=1)
ds = ds.reindex(np.random.permutation(ds.index))
train = ds


for x in cat_columns:
    train[x] = train[x].astype('category').cat.codes


s = StandardScaler()
for x in train.columns:
    if x not in label:
        train[x] = s.fit_transform(train[x].values.reshape(-1, 1)).astype('float64')


label_encoder = LabelEncoder()
train['y'] = label_encoder.fit_transform(train['y'])


train_features = train.values
train_label = train.pop('y').values


input_dim = train_features.shape[1]
model = keras.models.Sequential()
model.add(keras.layers.Dense(16, input_dim = input_dim, activation=tf.keras.layers.LeakyReLU()))
model.add(keras.layers.Dense(1, activation=tf.keras.layers.LeakyReLU()))
model.add(keras.layers.Dense(16,  activation=tf.keras.layers.LeakyReLU()))
model.add(keras.layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='Adagrad', loss='binary_crossentropy',
              metrics = 'binary_accuracy')


history = model.fit(train_features, train_label, epochs=10, validation_split=0.7)

val_acc = np.mean(history.history['val_binary_accuracy'])
results = model.evaluate(train_features, train_label)
print('\nLoss, Binary_accuracy: \n',(results))


style.use('dark_background')
pd.DataFrame(history.history).plot(figsize=(11, 7),linewidth=4)
plt.title('Binary Cross-entropy',fontsize=14, fontweight='bold')
plt.xlabel('Epochs',fontsize=13)
plt.ylabel('Metrics',fontsize=13)
plt.show()


