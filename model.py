import tensorflow as tf
import os
import cv2
import numpy as np


def load_data(dir_name):
    dir_list = os.listdir(dir_name)

    dir_list = filter(lambda x: x.find("jpg") >= 0, dir_list)

    def chr2index(char):
        char = ord(char)
        if char >= ord('0') and char <= ord('9'):
            return char - ord('0')
        return 10 + char - ord('A')

    X = []
    Y = []
    for dir_item in dir_list:
        file_path = os.path.join(dir_name, dir_item)

        img = cv2.imread(file_path)

        label = [0 for i in range(36 * 4)]
        label[chr2index(dir_item[0])] = 1
        label[chr2index(dir_item[2]) + 36] = 1
        label[chr2index(dir_item[3]) + 2 * 36] = 1
        label[chr2index(dir_item[4]) + 3 * 36] = 1
        # label = chr2index(dir_item[0])
        X.append(img)
        Y.append(label)

    return np.asarray(X), np.array(Y)





class Model():

    def __init__(self):
        pass

    def build_model(self):
        inputs = tf.keras.layers.Input(shape=[30, 120, 3])
        network = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        network = tf.keras.layers.MaxPool2D((2, 2))(network)
        network = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(network)
        network = tf.keras.layers.MaxPool2D((2, 2))(network)
        network = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(network)
        network = tf.keras.layers.MaxPool2D((2, 2))(network)
        network = tf.keras.layers.Flatten()(network)
        network = tf.keras.layers.Dense(1024, activation='relu')(network)
        network = tf.keras.layers.Dropout(0.2)(network)

        p1 = tf.keras.layers.Dense(36, activation="softmax")(network)
        p2 = tf.keras.layers.Dense(36, activation="softmax")(network)
        p3 = tf.keras.layers.Dense(36, activation="softmax")(network)
        p4 = tf.keras.layers.Dense(36, activation="softmax")(network)

        output = tf.keras.layers.Concatenate()([p1, p2, p3, p4])
        self.model = tf.keras.Model(inputs = [inputs], outputs=output)
        """
        
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(30, 120, 3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1024, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(36 * 4, activation='sigmoid')
            ]
        )
        """
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, X, label, epochs=32):
        self.model.fit(X, label, epochs=epochs)

    def predict(self, X):
        predictions = self.model.predict(X)
        print( predictions.shape)
        a = np.argmax(predictions[:, :36])
        b = np.argmax(predictions[:, 36 : 36 * 2])
        c = np.argmax(predictions[:, 36 * 2 : 36 * 3])
        d = np.argmax(predictions[:, 36 * 3: 36 * 4])
        return [a, b, c, d]

    def save_model(self, path):
        self.model.save_weights(path)

    def load_model(self, path):
        self.model.load_weights(path)

if __name__ == "__main__":
    X, Y = load_data('data')
    X = X / 255.0
    Y = np.asarray(Y)

    model = Model()
    model.build_model()
    #model.load_model("model.m")
    model.train(X, Y, 64)
    model.save_model("model.m")