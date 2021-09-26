import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.regularizers import l2


class EmotionBranch(Model):
    def __init__(self, backbone):
        super(EmotionBranch, self).__init__()
        self.backbone = backbone
        self.ga_pool = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(0.3)
        self.classifier = layers.Dense(8, kernel_regularizer=l2(1e-3),
                                       activation='softmax', name='emotion', dtype=tf.float32)

    def call(self, inputs, **kwargs):
        x = self.backbone(inputs)
        x = self.ga_pool(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class SupervisedModel(Model):
    def __init__(self, n_tiles, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_tiles = n_tiles
        self.backbone = ResNet50(include_top=False,
                                 input_shape=(224, 224, 3),
                                 weights=None,
                                 )
        self.emotion = EmotionBranch(backbone=self.backbone)

    def compile_(self, optimizer: Optimizer, loss_fns: dict, metrics: dict, loss_weights=None, run_eagerly=None):
        super(SupervisedModel, self).compile(run_eagerly=run_eagerly)
        self.optimizer = optimizer
        self.loss_fns = loss_fns
        self.metrics_dict = metrics
        self.loss_weights = loss_weights

    @property
    def metrics(self):
        metrics = []
        for m in self.metrics_dict.values():
            metrics += m
        return metrics

    def call(self, inputs, training=None, mask=None) -> dict:
        output = self.emotion(inputs)
        return output

    def train_step(self, data):
        x, y, weights = data

        with tf.GradientTape() as tape:
            outputs = self(x, training=True)
            loss = self.loss_fns['emotion'](y['emotion'], outputs, weights['emotion'])

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        self.metrics[0].update_state(y['emotion'], outputs)
        self.metrics[1].update_state(y['emotion'], outputs)
        emotion_acc = self.metrics[0].result()
        f1_acc = self.metrics[1].result()

        return {'loss': loss,
                'emotion_acc': emotion_acc,
                'f1': f1_acc}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.loss_fns['emotion'](y['emotion'], y_pred)
        self.metrics[0].update_state(y['emotion'], y_pred)
        self.metrics[1].update_state(y['emotion'], y_pred)
        acc = self.metrics[0].result()
        f1 = self.metrics[1].result()
        return {'loss': loss, 'acc': acc, 'f1': f1}


if __name__ == '__main__':
    # test model
    tf.random.set_seed(111)
    tf.keras.backend.clear_session()
    model = SupervisedModel(n_tiles=4)
    sample = tf.ones((1, 224, 224, 3))
    y = model(sample)
    print('')
