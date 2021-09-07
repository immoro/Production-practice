import tensorflow as tf
import tensorflow_probability as tfp

class DeepAR(tf.keras.models.Model):
    """
    DeepAR 模型
    """
    def __init__(self, lstm_units):
        super().__init__()
        # 注意，文章中使用了多层的 LSTM 网络，为了简单起见，本 demo 只使用一层
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        self.dense_mu = tf.keras.layers.Dense(1)
        self.dense_sigma = tf.keras.layers.Dense(1, activation='softplus')

    def call(self, inputs, initial_state=None):
        outputs, state_h, state_c = self.lstm(inputs, initial_state=initial_state)

        mu = self.dense_mu(outputs)
        sigma = self.dense_sigma(outputs)
        state = [state_h, state_c]

        return [mu, sigma, state]

def log_gaussian_loss(mu, sigma, y_true):
    """
    Gaussian 损失函数
    """
    return -tf.reduce_sum(tfp.distributions.Normal(loc=mu, scale=sigma).log_prob(y_true))

LSTM_UNITS = 16
EPOCHS = 5

# 实例化模型
model = DeepAR(LSTM_UNITS)

# 指定优化器
optimizer = tf.keras.optimizers.Adam()

# 使用 RMSE 衡量误差
rmse = tf.keras.metrics.RootMeanSquaredError()

# 定义训练步
def train_step(x, y):
    with tf.GradientTape() as tape:
        mu, sigma, _ = model(x)
        loss = log_gaussian_loss(mu, sigma, y)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    rmse(y, mu)

# 数据处理（略）
# train_data = do_something()

# 训练
for epoch in range(EPOCHS):
    for x, y in train_data:
        train_step(x, y)
    print('Epoch %d, RMSE %.4f' % (epoch + 1, rmse.result()))
    rmse.reset_states()
# ————————————————
# 原文链接：https://blog.csdn.net/weixin_45073190/article/details/104951504