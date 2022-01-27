import tensorflow as tf

class DiagonalToZero(tf.keras.constraints.Constraint):
    def __call__(self, w):
        """Set diagonal to zero"""
        q = tf.linalg.set_diag(w, tf.zeros(w.shape[0:-1]), name=None)
        return q

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a basket."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), stddev=1.)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VASP(tf.keras.Model):
    def __init__(self, num_words, latent=1024, hidden=1024, items_sampling=1.):
        """
        num_words             nr of items in dataset (size of tokenizer)
        latent                size of latent space
        hidden                size of hidden layers
        items_sampling        Large items datatsets can be very gpu memory consuming in EASE layer.
                              This coefficient reduces number of ease parametrs by taking only
                              fraction of items sorted by popularity as input for model.
                              Note: This coef should be somewhere around coverage@100 achieved by full
                              size model.
                              For ML20M this coef should be between 0.4888 (coverage@100 for full model)
                              and 1.0
                              For Netflix this coef should be between 0.7055 (coverage@100 for full
                              model) and 1.0
        """
        # super(tf.keras.Model, self).__init__()
        super().__init__()
        self.num_words = num_words
        self.sampled_items = int(num_words * items_sampling)

        assert self.sampled_items > 0
        assert self.sampled_items <= num_words

        self.s = self.sampled_items < num_words

        # ************* ENCODER ***********************
        self.encoder1 = tf.keras.layers.Dense(hidden)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.encoder2 = tf.keras.layers.Dense(hidden)
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.encoder3 = tf.keras.layers.Dense(hidden)
        self.ln3 = tf.keras.layers.LayerNormalization()
        self.encoder4 = tf.keras.layers.Dense(hidden)
        self.ln4 = tf.keras.layers.LayerNormalization()
        self.encoder5 = tf.keras.layers.Dense(hidden)
        self.ln5 = tf.keras.layers.LayerNormalization()
        self.encoder6 = tf.keras.layers.Dense(hidden)
        self.ln6 = tf.keras.layers.LayerNormalization()
        self.encoder7 = tf.keras.layers.Dense(hidden)
        self.ln7 = tf.keras.layers.LayerNormalization()

        # ************* SAMPLING **********************
        self.dense_mean = tf.keras.layers.Dense(latent,
                                                name="Mean")
        self.dense_log_var = tf.keras.layers.Dense(latent,
                                                    name="log_var")

        self.sampling = Sampling(name='Sampler')

        # ************* DECODER ***********************
        self.decoder1 = tf.keras.layers.Dense(hidden)
        self.dln1 = tf.keras.layers.LayerNormalization()
        self.decoder2 = tf.keras.layers.Dense(hidden)
        self.dln2 = tf.keras.layers.LayerNormalization()
        self.decoder3 = tf.keras.layers.Dense(hidden)
        self.dln3 = tf.keras.layers.LayerNormalization()
        self.decoder4 = tf.keras.layers.Dense(hidden)
        self.dln4 = tf.keras.layers.LayerNormalization()
        self.decoder5 = tf.keras.layers.Dense(hidden)
        self.dln5 = tf.keras.layers.LayerNormalization()

        self.decoder_resnet = tf.keras.layers.Dense(self.sampled_items,
                                                    activation='sigmoid',
                                                    name="DecoderR")
        self.decoder_latent = tf.keras.layers.Dense(self.sampled_items,
                                                    activation='sigmoid',
                                                    name="DecoderL")

        # ************* PARALLEL SHALLOW PATH *********

        self.ease = tf.keras.layers.Dense(
            self.sampled_items,
            activation='sigmoid',
            use_bias=False,
            kernel_constraint=DiagonalToZero(),  # critical to prevent learning simple identity
        )

    def call(self, x, training=None):
        sampling = self.s
        if sampling:
            sampled_x = x[:, :self.sampled_items]
            non_sampled = x[:, self.sampled_items:] * 0.
        else:
            sampled_x = x

        z_mean, z_log_var, z = self.encode(sampled_x)
        if training:
            d = self.decode(z)
            # Add KL divergence regularization loss.
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            self.add_loss(kl_loss)
            self.add_metric(kl_loss, name="kl_div")
        else:
            d = self.decode(z_mean)

        if sampling:
            d = tf.concat([d, non_sampled], axis=-1)

        ease = self.ease(sampled_x)

        if sampling:
            ease = tf.concat([ease, non_sampled], axis=-1)

        return d * ease

    def decode(self, x):
        e0 = x
        e1 = self.dln1(tf.keras.activations.swish(self.decoder1(e0)))
        e2 = self.dln2(tf.keras.activations.swish(self.decoder2(e1) + e1))
        e3 = self.dln3(tf.keras.activations.swish(self.decoder3(e2) + e1 + e2))
        e4 = self.dln4(tf.keras.activations.swish(self.decoder4(e3) + e1 + e2 + e3))
        e5 = self.dln5(tf.keras.activations.swish(self.decoder5(e4) + e1 + e2 + e3 + e4))

        dr = self.decoder_resnet(e5)
        dl = self.decoder_latent(x)

        return dr * dl

    def encode(self, x):
        e0 = x
        e1 = self.ln1(tf.keras.activations.swish(self.encoder1(e0)))
        e2 = self.ln2(tf.keras.activations.swish(self.encoder2(e1) + e1))
        e3 = self.ln3(tf.keras.activations.swish(self.encoder3(e2) + e1 + e2))
        e4 = self.ln4(tf.keras.activations.swish(self.encoder4(e3) + e1 + e2 + e3))
        e5 = self.ln5(tf.keras.activations.swish(self.encoder5(e4) + e1 + e2 + e3 + e4))
        e6 = self.ln6(tf.keras.activations.swish(self.encoder6(e5) + e1 + e2 + e3 + e4 + e5))
        e7 = self.ln7(tf.keras.activations.swish(self.encoder7(e6) + e1 + e2 + e3 + e4 + e5 + e6))

        z_mean = self.dense_mean(e7)
        z_log_var = self.dense_log_var(e7)
        z = self.sampling((z_mean, z_log_var))

        return z_mean, z_log_var, z


model = VASP(num_words = 20721, latent=2048, hidden=4096,items_sampling =0.5)
model.load_weights("VASP_ML20_1")

# import numpy as np
# e = np.random.choice([0,1],size =14409 ).reshape(1,-1)

def vasp_model(e):
	p = model.call(e)
	return p[0].numpy()