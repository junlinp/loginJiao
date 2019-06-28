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

    model = TFModel()
    model.build_model()
    #model.load_model("model.m")
    model.train(X, Y)
    #model.save_model("model.m")