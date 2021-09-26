import tensorflow as tf


def sum_losses(losses: dict, aggregate_ssl_losses=True):
    ssl_losses = list(losses.values())[1:]
    if aggregate_ssl_losses:
        ssl_loss = tf.divide(tf.add_n(ssl_losses), len(ssl_losses))
    else:
        ssl_loss = tf.add_n(ssl_losses)
    emotion_loss = losses['emotion']
    losses_sum = tf.add_n([ssl_loss, emotion_loss])
    return losses_sum


def geometric_loss(losses: dict, aggregate_ssl_losses, focused_loss_strategy):
    def prod_root(losses_: list):
        index = len(losses_)
        prod = tf.reduce_prod(losses_)
        root = tf.pow(prod, 1/index)
        return root

    ssl_losses = list(losses.values())[1:]
    if aggregate_ssl_losses:
        ssl_loss = tf.divide(tf.add_n(ssl_losses), len(ssl_losses))
    else:
        ssl_loss = tf.reduce_prod(ssl_losses)
    emotion_loss = losses['emotion']
    losses_root = prod_root([ssl_loss, emotion_loss])
    if focused_loss_strategy:
        losses_root *= tf.sqrt(losses_root)
    return losses_root
