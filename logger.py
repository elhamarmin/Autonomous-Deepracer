import datetime
import tensorflow as tf
from keras import callbacks , models

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

class CustomMetrics(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Log additional metrics
        logs = logs or {}
        logs['custom_metric'] = custom_metric_value
        tf.summary.scalar('custom_metric', data=custom_metric_value, step=epoch)

def log_gradients(epoch, logs):
    with tf.GradientTape() as tape:
        predictions = model(x_train)
        loss = loss_fn(y_train, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    for grad, var in zip(gradients, model.trainable_variables):
        tf.summary.histogram(var.name + '/gradient', data=grad, step=epoch)

def log_weights(epoch, logs):
    for layer in model.layers:
        for weight in layer.trainable_weights:
            tf.summary.histogram(weight.name, data=weight, step=epoch)

def log_learning_rate(epoch, logs):
    lr = model.optimizer.lr
    tf.summary.scalar('learning rate', data=lr, step=epoch)

def log_reward(episode, reward):
    tf.summary.scalar('reward', data=reward, step=episode)

reward_callback = callbacks.LambdaCallback(on_epoch_end=log_reward)
lr_callback = callbacks.LambdaCallback(on_epoch_end=log_learning_rate)
weights_callback = callbacks.LambdaCallback(on_epoch_end=log_weights)
gradient_callback = callbacks.LambdaCallback(on_epoch_end=log_gradients)

custom_metrics_callback = CustomMetrics()
callbacks = [
    tensorboard_callback,
    custom_metrics_callback,
    gradient_callback,
    weights_callback,
    lr_callback,
    reward_callback
]
