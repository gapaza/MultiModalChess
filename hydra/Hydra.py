from keras import layers

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

        # --> Output Heads
        self.move_prediction_head = MovePrediction()
        self.move_mask_prediction_head = MoveMaskPrediction()

    def __call__(self, board_inputs, move_inputs, mask=None):

        # --> Board Embedding
        board_embedding = self.board_attention(board_inputs)

        # --> Move Embedding
        move_embedding = self.move_embedding(move_inputs)

        # --> Combine Board and Move Embeddings
        encoder_inputs = self.modality_fusion(board_embedding, move_embedding)

        # --> Encoder Stack
        encoder_outputs = self.encoder(encoder_inputs)

        # --> Output Heads
        split_idx = config.vt_num_patches
        encoder_board_output = encoder_outputs[:, :split_idx, :]
        encoder_move_output = encoder_outputs[:, split_idx:, :]
        output = []
        if self.mode == 'pretrain':
            output = self.move_mask_prediction_head(encoder_move_output)
        elif self.mode == 'predict':
            output = self.move_prediction_head(encoder_outputs)
        return output

    def call2(self, board_inputs, move_inputs, mask=None):

        # 1. Move Embedding
        move_embedding = self.move_embedding(move_inputs)

        # 2. Board Embedding
        board_embedding = self.board_embedding(board_inputs)

        # 3. Combine Board and Move Embeddings
        combined_embedding = self.modality_fusion(board_embedding, move_embedding)

        # 4. Positional Embedding
        combined_positional_embedding = self.positional_embedding(combined_embedding)

        # 5. Visual Encoder
        encoder_outputs = self.visual_encoder(combined_positional_embedding)

        # 6. Output Heads
        split_idx = config.vt_num_patches
        encoder_board_output = encoder_outputs[:, :split_idx, :]
        encoder_move_output = encoder_outputs[:, split_idx:, :]
        output = []
        if self.mode == 'pretrain':
            output = self.move_mask_prediction_head(encoder_move_output)
        elif self.mode == 'predict':
            output = self.move_prediction_head(encoder_outputs)
        return output
















