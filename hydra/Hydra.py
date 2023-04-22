from keras import layers

# --> Custom Layers
from hydra.layers.MoveEmbedding import MoveEmbedding
from hydra.layers.BoardAttention import BoardAttention
from hydra.layers.ModalityFusion import ModalityFusion
from hydra.layers.Encoder import Encoder

# --> Output Heads
from hydra.heads.MovePrediction import MovePrediction
from hydra.heads.MoveMaskPrediction import MoveMaskPrediction

# --> Config


class Hydra(layers.Layer):

    def __init__(self, *args, **kwargs):
        super(Hydra, self).__init__(*args, **kwargs)
        self.mode = 'pretrain'

        # --> Move Embedding
        self.move_embedding = MoveEmbedding()

        # --> Board Embedding
        self.board_embedding = BoardAttention()

        # --> Modality Fusion
        self.modality_fusion = ModalityFusion()

        # --> Encoder
        self.encoder = Encoder()

        # --> Output Heads
        self.move_prediction_head = MovePrediction()
        self.move_mask_prediction_head = MoveMaskPrediction()

    def __call__(self, board_inputs, move_inputs, mask=None):

        # --> Board Embedding
        board_embedding = self.board_embedding(board_inputs)

        # --> Move Embedding
        move_embedding = self.move_embedding(move_inputs)

        # --> Combine Board and Move Embeddings
        encoder_inputs = self.modality_fusion(board_embedding, move_embedding)

        # --> Encoder Stack
        encoder_outputs = self.encoder(encoder_inputs)

        # --> Output Heads
        split_idx = 16
        encoder_board_output = encoder_outputs[:, :split_idx, :]
        encoder_move_output = encoder_outputs[:, split_idx:, :]
        output = []
        if self.mode == 'pretrain':
            output = self.move_mask_prediction_head(encoder_move_output)
        elif self.mode == 'predict':
            output = self.move_prediction_head(encoder_outputs)
        return output


















