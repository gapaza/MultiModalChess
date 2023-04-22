from keras import layers
import tensorflow as tf
from hydra import config
import keras
import os
from keras.utils import plot_model
import math
from hydra.board_attention.ShiftedPatchTokenization import ShiftedPatchTokenization
from hydra.board_attention.PatchEncoder import PatchEncoder
from hydra.board_attention.MultiHeadAttentionLSA import MultiHeadAttentionLSA

# AUGMENTATION
IMAGE_SIZE = 8
PATCH_SIZE = 2
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2

# ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 64

INPUT_SHAPE = (8, 8, 12)


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


# Build the diagonal attention mask
diag_attn_mask = 1 - tf.eye(NUM_PATCHES)
diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)


def create_vit_classifier(vanilla=False):
    inputs = layers.Input(shape=(8, 8, 12))

    # Create patches.
    (tokens, _) = ShiftedPatchTokenization(vanilla=vanilla)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder()(tokens)

    # Create multiple layers of the Transformer block.
    for _ in range(config.visual_transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        if not vanilla:
            attention_output = MultiHeadAttentionLSA(
                num_heads=config.visual_transformer_heads, key_dim=PROJECTION_DIM, dropout=0.1
            )(x1, x1, attention_mask=diag_attn_mask)
        else:
            attention_output = layers.MultiHeadAttention(
                num_heads=config.visual_transformer_heads, key_dim=PROJECTION_DIM, dropout=0.1
            )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=config.visual_transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])



    model = tf.keras.Model(inputs=inputs, outputs=encoded_patches)

    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer, jit_compile=False)

    model.summary()
    model_img_file = os.path.join(config.board_attention_dir, 'visual_transformer.png')
    plot_model(model, to_file=model_img_file, show_shapes=True, show_layer_names=True, expand_nested=True)



if __name__ == "__main__":
    create_vit_classifier()







