import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.regularizers import l2
from loss import geometric_loss, sum_losses


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


class SSLBranchBase(Model):
    def __init__(self, backbone):
        super(SSLBranchBase, self).__init__()
        self.backbone = backbone
        self.conv1 = layers.Conv2D(128, (1, 1), activation='linear')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(256, (3, 3), activation='linear')
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.conv3 = layers.Conv2D(256, (3, 3), activation='linear')
        self.bn3 = layers.BatchNormalization()
        self.relu3 = layers.ReLU()
        self.pool = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(0.2)

    def call(self, inputs, **kwargs):
        x = self.backbone(inputs)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.dropout(self.pool(x))
        return x


class SSL(Model):
    def __init__(self, n_tiles, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_tiles = n_tiles
        self.backbone = ResNet50(include_top=False,
                                 input_shape=(224, 224, 3),
                                 weights=None,
                                 )
        self.emotion = EmotionBranch(backbone=self.backbone)
        self.ssl_branch_base = SSLBranchBase(backbone=self.backbone)
        self.base_heads = {f'x{i + 1}': layers.Dense(16, activation='relu') for i in range(self.n_tiles)}
        self.puzzle_heads = {f'puzzle_{i + 1}': layers.Dense(self.n_tiles,
                                                             activation='softmax',
                                                             name=f'puzzle_{i + 1}', dtype='float32')
                             for i in range(self.n_tiles)}

    def compile_(self, optimizer: Optimizer, loss_fns: dict, metrics: dict, loss_weights=None, run_eagerly=None):
        super(SSL, self).compile(run_eagerly=run_eagerly)
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
        emotion = self.emotion(inputs)
        ssl_base_out = self.ssl_branch_base(inputs)

        outputs = {'emotion': emotion}
        for i, (head_name, layer) in enumerate(self.base_heads.items()):
            ssl_base_output = layer(ssl_base_out)

            puzzle_head_name = list(self.puzzle_heads)[i]
            puzzle_layer = self.puzzle_heads[puzzle_head_name]
            outputs[f'puzzle_{i + 1}'] = puzzle_layer(ssl_base_output)

        return outputs

    def train_step(self, data):
        x, y, weights = data

        with tf.GradientTape() as tape:
            outputs = self(x, training=True)
            losses = {}
            for head, output in outputs.items():
                losses[head] = self.loss_fns[head](y[head], outputs[head], weights[head])
            losses['emotion'] *= 30
            merged_loss = geometric_loss(losses, aggregate_ssl_losses=True, focused_loss_strategy=False)

        trainable_variables = self.trainable_variables
        self.optimizer.minimize(merged_loss, trainable_variables, tape=tape)

        self.metrics[0].update_state(y['emotion'], outputs['emotion'])
        self.metrics[1].update_state(y['emotion'], outputs['emotion'])
        self.metrics[2].update_state(y['puzzle_1'], outputs['puzzle_1'])
        emotion_acc = self.metrics[0].result()
        f1_acc = self.metrics[1].result()
        puzzle_acc = self.metrics[2].result()

        return {'loss': merged_loss,
                'emotion_loss': losses['emotion'],
                'emotion_acc': emotion_acc,
                'f1': f1_acc,
                'puzzle_acc': puzzle_acc}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.loss_fns['emotion'](y['emotion'], y_pred['emotion'])
        self.metrics[0].update_state(y['emotion'], y_pred['emotion'])
        self.metrics[1].update_state(y['emotion'], y_pred['emotion'])
        self.metrics[2].update_state(y['puzzle_1'], y_pred['puzzle_1'])
        acc = self.metrics[0].result()
        f1 = self.metrics[1].result()
        puzzle_acc = self.metrics[2].result()
        return {'loss': loss, 'acc': acc, 'f1': f1, 'puzzle_acc': puzzle_acc}


if __name__ == '__main__':
    # test model
    tf.random.set_seed(111)
    tf.keras.backend.clear_session()
    model = SSL(n_tiles=4)
    sample = tf.ones((1, 224, 224, 3))
    y = model(sample)
    print('')
