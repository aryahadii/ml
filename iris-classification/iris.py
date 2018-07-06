from __future__ import absolute_import, division, print_function

import os

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe

EPOCHS = 200
LEARNING_RATE = 0.01
TEST_DATASET_URL = "http://download.tensorflow.org/data/iris_test.csv"
TRAIN_DATASET_URL = "http://download.tensorflow.org/data/iris_training.csv"


def parse_csv(line):
    defaults = [[0.], [0.], [0.], [0.], [0]]
    parsed_line = tf.decode_csv(line, defaults)

    features = tf.reshape(parsed_line[:-1], shape=(4, ))
    label = tf.reshape(parsed_line[-1], shape=())
    return features, label


def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def gradient(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)


def load_dataset(dataset_url):
    dataset_file = tf.keras.utils.get_file(
        fname=os.path.basename(dataset_url), origin=dataset_url)

    dataset = tf.data.TextLineDataset(dataset_file)
    return dataset.skip(1).map(parse_csv).batch(32).shuffle(buffer_size=1000)


def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation="relu", input_shape=(4, )),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(3)
    ])


def train_model(train_dataset, model):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)

    train_loss_results = []
    train_accuracy_results = []

    for epoch in range(EPOCHS):
        avg_loss = tfe.metrics.Mean()
        accuracy = tfe.metrics.Accuracy()

        for x, y in train_dataset:
            grads = gradient(model, x, y)
            optimizer.apply_gradients(
                zip(grads, model.variables),
                global_step=tf.train.get_or_create_global_step())

            avg_loss(loss(model, x, y))
            accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

        train_loss_results.append(avg_loss.result())
        train_accuracy_results.append(accuracy.result())

        if epoch % 50 == 0:
            print("loss: {:.3f}, accuracy: %{:.2f}".format(
                avg_loss.result(),
                accuracy.result() * 100))

    return train_loss_results, train_accuracy_results


def plot_metrics(loss, accuracy):
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle("Metrics")

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(loss)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].plot(accuracy)
    axes[1].set_xlabel("Epoch")

    plt.show()


def calculate_test_accuracy(test_dataset, model):
    test_accuracy = tfe.metrics.Accuracy()
    for x, y in test_dataset:
        prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)

    return test_accuracy.result()


def main():
    tf.enable_eager_execution()

    # Train model
    train_dataset = load_dataset(TRAIN_DATASET_URL)
    model = create_model()
    loss, accuracy = train_model(train_dataset, model)
    plot_metrics(loss, accuracy)

    # Test model
    test_dataset = load_dataset(TEST_DATASET_URL)
    print("Test accuracy: %{:.2f}".format(
        calculate_test_accuracy(test_dataset, model) * 100))


if __name__ == "__main__":
    main()
