from keras import layers
from keras_nlp.layers import TransformerDecoder

# --> Custom Layers
from hydra.layers.MoveEmbedding import MoveEmbedding
from hydra.layers.BoardEmbedding import BoardEmbedding
from hydra.layers.ModalityFusion import ModalityFusion
from hydra.layers.BoardAttention import BoardAttention
from hydra.layers.Encoder import Encoder
from hydra.layers.VisualEncoder import VisualEncoder
from hydra.layers.PositionalEmbedding import PositionalEmbedding

# --> Output Heads
from hydra.heads.MovePrediction import MovePrediction
from hydra.heads.MoveMaskPrediction import MoveMaskPrediction

# --> Config
from hydra import config



class Hydra(layers.Layer):

    def __init__(self, *args, **kwargs):
        super(Hydra, self).__init__(*args, **kwargs)
        self.mode = 'pretrain'

        # --> Move Embedding
        self.move_embedding = MoveEmbedding(positional=False)
        self.decoder_move_embedding = MoveEmbedding(positional=True)

        # --> Board Embedding
        self.board_embedding = BoardEmbedding()
        self.board_attention = BoardAttention()

        # --> Modality Fusion
        self.modality_fusion = ModalityFusion()

        # --> Position Embeddings
        self.positional_embedding = PositionalEmbedding()

        # --> Encoders
        self.encoder = Encoder()
        self.visual_encoder = VisualEncoder()
        # self.visual_encoder_2 = VisualEncoder()

        # --> Decoders
        self.decoder = TransformerDecoder(config.encoder_dense_dim, config.encoder_heads)

        # --> Output Heads
        self.autoregressive_head = layers.Dense(config.vocab_size, activation='softmax')

    def __call__(self, board_inputs, encoder_move_inputs, decoder_move_inputs, mask=None):

        # 1. Move Embeddings
        encoder_move_embedding = self.move_embedding(encoder_move_inputs)

        # 2. Board Embedding
        board_embedding = self.board_embedding(board_inputs)

        # 3. Combine Board and Move Embeddings
        combined_embedding = self.modality_fusion(board_embedding, encoder_move_embedding)

        # 4. Positional Embedding
        combined_positional_embedding = self.positional_embedding(combined_embedding)

        # 5. Visual Encoder
        encoder_outputs = self.visual_encoder(combined_positional_embedding)

        # 6. Decoder
        decoder_move_embedding = self.decoder_move_embedding(decoder_move_inputs)
        decoder_outputs = self.decoder(decoder_move_embedding, encoder_outputs)
        decoder_outputs = self.decoder_dropout(decoder_outputs)

        # 7. Output Heads
        output = self.autoregressive_head(decoder_outputs)
        return output











