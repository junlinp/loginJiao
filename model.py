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
        network = tf.keras.layers.BatchNormalization()(network)
        network = tf.keras.layers.MaxPool2D((2, 2))(network)
        network = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(network)
        network = tf.keras.layers.BatchNormalization()(network)
        network = tf.keras.layers.MaxPool2D((2, 2))(network)

        network = tf.keras.layers.Flatten()(network)

        p1 = tf.keras.layers.Dense(128, activation='relu')(network)
        p1 = tf.keras.layers.Dropout(0.2)(p1)
        p2 = tf.keras.layers.Dense(128, activation='relu')(network)
        p2 = tf.keras.layers.Dropout(0.2)(p2)
        p3 = tf.keras.layers.Dense(128, activation='relu')(network)
        p3 = tf.keras.layers.Dropout(0.2)(p3)
        p4 = tf.keras.layers.Dense(128, activation='relu')(network)
        p4 = tf.keras.layers.Dropout(0.2)(p4)

        p1 = tf.keras.layers.Dense(36, activation="softmax")(p1)
        p2 = tf.keras.layers.Dense(36, activation="softmax")(p2)
        p3 = tf.keras.layers.Dense(36, activation="softmax")(p3)
        p4 = tf.keras.layers.Dense(36, activation="softmax")(p4)

        self.model1 = tf.keras.Model(inputs = [inputs], outputs=p1)
        self.model2 = tf.keras.Model(inputs = [inputs], outputs=p2)
        self.model3 = tf.keras.Model(inputs = [inputs], outputs=p3)
        self.model4 = tf.keras.Model(inputs = [inputs], outputs=p4)

        self.model1.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model1.summary()
        self.model2.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        self.model2.summary()
        self.model3.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        self.model3.summary()
        self.model4.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        self.model4.summary()

    def train(self, X, label, X_valid=None, Y_valid=None,epochs=32):
        label1 = label[:, : 36]
        label2 = label[:, 36 : 36 * 2]
        label3 = label[:, 36 * 2: 36 * 3]
        label4 = label[:, 36 * 3: 36 * 4]

        Y_valid1 = Y_valid[:, : 36]
        Y_valid2 = Y_valid[:, 36 : 36 * 2]
        Y_valid3 = Y_valid[:, 36 * 2: 36 * 3]
        Y_valid4 = Y_valid[:,36 * 3 : 36 * 4]
        if X_valid is None or Y_valid is None:
            self.model1.fit(X, label1, epochs=epochs, batch_size=64)
            self.model2.fit(X, label2, epochs=epochs, batch_size=64)
            self.model3.fit(X, label3, epochs=epochs, batch_size=64)
            self.model4.fit(X, label4, epochs=epochs, batch_size=64)
        else:
            self.model1.fit(X, label1, epochs=epochs, batch_size=64, validation_data=(X_valid, Y_valid1))
            self.model2.fit(X, label2, epochs=epochs, batch_size=64, validation_data=(X_valid, Y_valid2))
            self.model3.fit(X, label3, epochs=epochs, batch_size=64, validation_data=(X_valid, Y_valid3))
            self.model4.fit(X, label4, epochs=epochs, batch_size=64, validation_data=(X_valid, Y_valid4))

    def predict(self, X):
        a = self.model1.predict(X)
        b = self.model2.predict(X)
        c = self.model3.predict(X)
        d = self.model4.predict(X)
        return [a, b, c, d]

    def evaluate(self, X, Y):
        pass

    def save_model(self, path):
        self.model1.save_weights(path + "1")
        self.model2.save_weights(path + "2")
        self.model3.save_weights(path + "3")
        self.model4.save_weights(path + "4")

    def load_model(self, path):
        self.model1.load_weights(path + "1")
        self.model2.load_weights(path + "2")
        self.model3.load_weights(path + "3")
        self.model4.load_weights(path + "4")

if __name__ == "__main__":
    X, Y = load_data('data')
    X = X / 255.0
    Y = np.asarray(Y)
    X_valid, Y_valid = load_data('valid')
    X_valid = X_valid / 255.0

    model = Model()
    model.build_model()
    #model.load_model("model.m")
    model.train(X, Y, X_valid, Y_valid, 16)
    model.save_model("model.m")
    predict_input = np.array([X[0, :, :, :]])

    print( model.predict( predict_input) )
    print(Y[0, :])
