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





class TFModel():
    def __init__(self):
        pass

    def build_model(self):
        self.x_input = tf.placeholder(tf.float32, [None, 30, 120, 3])
        self.y_input = tf.palceholder(tf.float32, [None, 36 * 4])

        conv1 = tf.nn.conv2d(self.x_input, 32, [3, 3])
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.conv2d(conv1, 32, [3, 3])
        conv1 = tf.nn.relu(conv1)
        max_pool1 = tf.nn.max_pool2d(conv1, [2, 2], [2, 2])

        conv2 = tf.nn.conv2d(max_pool1, 64, [3, 3])
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.conv2d(conv2, 64, [3, 3])
        conv2 = tf.nn.relu(conv2)
        max_pool2 = tf.nn.max_pool2d(conv2, [2, 2], [2, 2])

        reshape = tf.nn.reshape(max_pool2, [-1, 30 * 64 * 8])

        fc1 = tf.layers.dense(reshape, 1024)
        fc1 = tf.layers.dense(fc1, 36)

        fc2 = tf.layers.dense(reshape, 1024)
        fc2 = tf.layers.dense(fc2, 36)

        fc3 = tf.layers.dense(reshape, 1024)
        fc3 = tf.layers.dense(fc3, 36)

        fc4 = tf.layers.dense(reshape, 1024)
        fc4 = tf.layers.dense(fc4, 36)

        concat = tf.concat([fc1, fc2, fc3, fc4], axis = 1)
        self.cross_entropy = tf.reduce_mean(
           tf.nn.softmax_cross_entropy_with_logits(logits = concat, labels = self.y_input)
        )

        self.optimiser = tf.train.AdamOptimizer(0.01).minimize(self.cross_entropy)
        correct_prediction = \
            tf.cast(tf.equal(tf.argmax(fc1, 1), tf.argmax(self.y_input[:36], 1)), tf.int32) * \
            tf.cast(tf.equal(tf.argmax(fc2, 1), tf.argmax(self.y_input[36 : 2 * 36])), tf.int32) *\
            tf.cast(tf.equal(tf.argmax(fc3, 1), tf.argmax(self.y_input[36 * 2, 36 * 3])), tf.int32) *\
            tf.cast(tf.euqal(tf.argmax(fc4, 1), tf.argmax(self.y_input[36 * 3, 36 * 4])), tf.int32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)

    def train(self, X, Y, epochs = 32):

        for i in range(epochs):
            _, accurary,loss = self.sess.run([self.optimiser, self.accuracy, self.cross_entropy], feed_dict={'x_input' : X, 'y_input' : Y})
            print("EPOCHS {} -- Accurary: {}".format(i + 1, accurary))



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
