import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

dataset = pd.read_csv('../cvd/input/cvd.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

pca = PCA(n_components=0.95)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
ann.add(Dropout(0.3))

ann.add(tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
ann.add(Dropout(0.3))

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

ann.fit(X_train, y_train, batch_size=32, epochs=100, class_weight=class_weights, validation_data=(X_test, y_test), callbacks=[early_stopping])

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

ann.save('cvd_model_with_pca.h5')
print("Model saved to 'cvd_model_with_pca.h5'.")
