import tensorflow as tf
from tensorflow.keras import layers, models

class DenseBlock3D(tf.keras.Model):
    def __init__(self, growth_rate, num_layers, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        self.layers_list = []

        for i in range(num_layers):
            self.layers_list.append([
                layers.LeakyReLU(alpha),
                layers.Conv3D(growth_rate, kernel_size=3, padding='same', use_bias=False)
            ])

    def call(self, x):
        features = [x]
        for relu, conv in self.layers_list:
            out = relu(tf.concat(features, axis=-1))
            out = conv(out)
            features.append(out)
        return tf.concat(features, axis=-1)

class UNet3D_Dense_Cond(tf.keras.Model):
    def __init__(self, in_channels=1, base_channels=32, growth_rate=16,
                 num_blocks=6, layers_per_block=4, num_classes=1, cond_values=None):
        super().__init__()
        self.num_classes = num_classes
        self.cond_values = cond_values if cond_values is not None else [1.0] * num_classes

        self.initial_conv = layers.Conv3D(base_channels, kernel_size=3, padding='same', use_bias=False)

        # Encoder
        self.encoder_blocks = []
        self.pool_layers = []
        channels = base_channels
        for _ in range(num_blocks):
            block = DenseBlock3D(growth_rate, layers_per_block)
            self.encoder_blocks.append(block)
            self.pool_layers.append(layers.MaxPool3D(pool_size=2))
            channels += growth_rate * layers_per_block

        self.bottleneck = DenseBlock3D(growth_rate, layers_per_block)
        bn_channels = channels + growth_rate * layers_per_block

        # Decoder
        self.up_convs = []
        self.decoder_blocks = []
        dec_channels = bn_channels
        for _ in range(num_blocks):
            up = layers.Conv3DTranspose(channels, kernel_size=2, strides=2, padding='same', use_bias=False)
            self.up_convs.append(up)
            self.decoder_blocks.append(DenseBlock3D(growth_rate, layers_per_block))
            dec_channels = channels * 2 + growth_rate * layers_per_block
            channels -= growth_rate * layers_per_block

        self.final_conv = layers.Conv3D(1, kernel_size=1)
        self.sigmoid = layers.Activation('sigmoid')

    def make_cond_map(self, x, class_idx):
        val = self.cond_values[class_idx]
        cond_shape = tf.shape(x)
        cond = tf.fill([cond_shape[0], cond_shape[1], cond_shape[2], cond_shape[3], 1], val)
        return cond

    def call(self, x, class_idx):
        cond = self.make_cond_map(x, class_idx)
        x = tf.concat([x, cond], axis=-1)
        x = self.initial_conv(x)

        skips = []
        for pool, block in zip(self.pool_layers, self.encoder_blocks):
            cond = self.make_cond_map(x, class_idx)
            x = tf.concat([x, cond], axis=-1)
            x = block(x)
            skips.append(x)
            x = pool(x)

        cond = self.make_cond_map(x, class_idx)
        x = tf.concat([x, cond], axis=-1)
        x = self.bottleneck(x)

        for up, block, skip in zip(self.up_convs, self.decoder_blocks, reversed(skips)):
            x = up(x)
            # Padding not implemented here; assumes perfect spatial alignment
            x = tf.concat([skip, x], axis=-1)
            cond = self.make_cond_map(x, class_idx)
            x = tf.concat([x, cond], axis=-1)
            x = block(x)

        x = self.final_conv(x)
        return self.sigmoid(x)

# Example usage
if __name__ == "__main__":
    model = UNet3D_Dense_Cond(in_channels=1, num_classes=4, cond_values=[0.2, 0.4, 0.6, 0.8])
    input_tensor = tf.random.normal([1, 64, 128, 128, 1])  # TensorFlow uses channel-last format
    output = model(input_tensor, class_idx=2)
    print("Output shape:", output.shape)