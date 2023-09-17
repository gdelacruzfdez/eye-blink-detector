import torch
import torchvision.models as models
from einops import rearrange
from torch import nn

from model.timeseries_transformer import TimeSeriesTransformer


class CNNTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.task = config.get("task")
        self.num_classes = 3 if self.task == "blink_completeness_detection" else 2
        self.num_frames = config["num_frames"]
        self.optimizer_type = config.get('optimizer_type')
        self.dim = config["dim"]
        self.batch_size = config["batch_size"]
        self.encoder_layers = config["encoder_layers"]
        self.heads = config["heads"]
        self.in_channels = config["in_channels"]
        self.dropout = config["dropout"]
        self.emb_dropout = config["emb_dropout"]
        self.transformer_lr = config["transformer_lr"]
        self.cnn_lr = config["cnn_lr"]
        self.encoder_layer_type = config["encoder_layer_type"]
        self.use_lstm = config.get("use_lstm")
        self.use_siamese = config.get("use_siamese")
        self.use_vit = config.get("use_vit")
        self.use_weights = config.get("use_weights")

        self.softmax = nn.Softmax(dim=1)

        self.cnn_model = models.efficientnet_b2(pretrained=True)
        num_ftrs = self.cnn_model.classifier[1].in_features
        self.cnn_model.classifier[1] = nn.Linear(num_ftrs, self.dim)

        self.sequence_model = TimeSeriesTransformer(self.dim, self.num_frames, True,
                                                    dim_val=self.dim, n_heads=self.heads,
                                                    n_encoder_layers=self.encoder_layers,
                                                    window_size=self.num_frames,
                                                    encoder_layer_type=self.encoder_layer_type,
                                                    dim_feedforward_encoder=self.dim * 4)

        self.mlp_head = nn.Sequential(
            nn.MaxPool2d(kernel_size=(self.num_frames + 1, 1)),
            nn.Flatten(start_dim=1),  # flatten the tensor
            nn.Linear(self.dim, self.num_classes)
        )

        self.cnn_cache = None

    def forward(self, x):
        print('X size', x.size())
        if self.cnn_cache is None:
            # This means it's the first iteration

            # Process all images in the initial batch
            cnn_output = self.cnn_model(x)

            # Cache results of the first 32 images
            self.cnn_cache = cnn_output[:-self.batch_size].detach()  # Exclude the last image

        else:
            # This means we're in a subsequent iteration

            # Process only the new image (last image in x)
            new_cnn_output = self.cnn_model(x[-self.batch_size:])  # Process the last image
            # Shift the cached results and add the new result at the end
            self.cnn_cache = torch.cat((self.cnn_cache[self.batch_size:], new_cnn_output),
                                       dim=0).detach()  # Shift and append the new output

            # Combine cached results with the new result
            cnn_output = torch.cat((self.cnn_cache, new_cnn_output), dim=0)

        x = self.create_sliding_window_padded(cnn_output, self.num_frames)
        x = self.sequence_model(x)
        x = self.mlp_head(x)
        return self.softmax(x)

    def create_sliding_window_padded(self, vector, window_size):
        # Initialize output tensor
        output = torch.zeros((self.batch_size, window_size + 1, self.dim), dtype=vector.dtype, device=vector.device)

        for i in range(self.batch_size):
            output[i] = vector[i:i + window_size + 1]

        return output


config_transformer = {
    'image_size': 64, 'num_frames': 32, 'batch_size': 5, 'dim': 128,
    'encoder_layers': 4, 'heads': 8, 'cnn_lr': 0,
    'transformer_lr': 0, 'lr': 0, 'data_augmentation': True, 'train_mode': 'train',
    "num_classes": 2, "in_channels": 3, "dropout": 0, "emb_dropout": 0,
    "version": "eye_trans",
    "encoder_layer_type": "local",
    'optimizer_type': 'radam',
    "use_lstm": False,
    "use_siamese": False,
    "task": "blink_detection"
}


def get_blink_predictor() -> CNNTransformer:
    model = CNNTransformer(config_transformer)
    checkpoint = torch.load(r"C:\Users\Gonzalo\PycharmProjects\eye-blink-detector\model.ckpt",
                            map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
