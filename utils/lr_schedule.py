import numpy as np
import tensorflow as tf

def static_scheduler_warm_up(epoch_num):
    def scheduler_func(epoch_idx, lr):
        if epoch_idx == 5:
            lr = lr * 10
        if epoch_idx == epoch_num // 2:
            lr = lr / 10
        if epoch_idx == 3 * epoch_num // 4:
            lr = lr / 10
        return lr

    return scheduler_func


def exp_decay_scheduler(epoch_num, end_rate):
    end_exp = np.log(end_rate)
    step_exp = end_exp / epoch_num

    def scheduler_func(epoch_idx, lr):
        return lr * np.exp(step_exp)

    return scheduler_func


def circle_linear_schedule(epoch_num, high_rate, low_rate, iter=4):
    circle_step = epoch_num / iter
    circle_step = (high_rate - low_rate) / circle_step

    def scheduler_func(epoch_idx, lr):
        if epoch_idx // circle_step == 0:
            lr = high_rate
        else:
            lr = lr - circle_step
        return lr


def static_learning_rate(epoch_num, epoch_idx, init_rate):
    if epoch_idx == 2 / 5 * epoch_num:
        return init_rate / 5
    if epoch_idx == 3 / 5 * epoch_num:
        return init_rate / 25
    if epoch_idx == 4 / 5 * epoch_num:
        return init_rate / 125
    return None


def compute_grad(model, x, y, loss_fun):

    with tf.GradientTape(persistent=True) as tape:
        # Forward pass
        y_pre = model.predict_on_batch(x[:128])
        loss = loss_fun(y, y_pre)
        cur_grad = tape.gradient(loss, [layer.trainable_variables for layer in model.layers])
    return cur_grad