import torch
import torchvision.models as models
from torch import nn
from model.timeseries_transformer import TimeSeriesTransformer
import os



class CNNTransformer(nn.Module):
    def __init__(self,
                 num_frames=32,
                 batch_size=5,
                 dim=128,
                 encoder_layers=4,
                 heads=8,
                 encoder_layer_type="local",
                 task="blink_completeness_detection"
                 ):
        super().__init__()

        self.num_frames = num_frames
        self.task = task
        self.num_classes = 3 if self.task == "blink_completeness_detection" else 2
        self.dim = dim
        self.batch_size = batch_size
        self.encoder_layers = encoder_layers
        self.heads = heads
        self.encoder_layer_type = encoder_layer_type

        self.softmax = nn.Softmax(dim=1)

        self.cnn_model = models.efficientnet_b2()
        num_ftrs = self.cnn_model.classifier[1].in_features
        self.cnn_model.classifier[1] = nn.Linear(num_ftrs, self.dim)

        self.sequence_model = TimeSeriesTransformer(self.dim, num_frames, True,
                                                    dim_val=self.dim, n_heads=self.heads,
                                                    n_encoder_layers=self.encoder_layers,
                                                    window_size=num_frames,
                                                    encoder_layer_type=self.encoder_layer_type,
                                                    dim_feedforward_encoder=self.dim * 4)

        self.mlp_head = nn.Sequential(
            nn.MaxPool2d(kernel_size=(num_frames + 1, 1)),
            nn.Flatten(start_dim=1),  # flatten the tensor
            nn.Linear(self.dim, self.num_classes)
        )

        self.cnn_cache = None

    def forward(self, x):
        if self.cnn_cache is None:
            # This means it's the first iteration

            # Process all images in the initial batch
            cnn_output = self.cnn_model(x)

            # Cache results of the first 32 images
            self.cnn_cache = cnn_output.detach()

        else:
            # This means we're in a subsequent iteration

            # Process only the new image (last image in x)
            new_cnn_output = self.cnn_model(x[-self.batch_size:])  # Process the last image
            # Shift the cached results and add the new result at the end
            cnn_output = torch.cat((self.cnn_cache[self.batch_size:], new_cnn_output),
                                       dim=0)  # Shift and append the new output

            # Combine cached results with the new result
            self.cnn_cache = cnn_output.detach()

        x = self.sliding_window(cnn_output, self.batch_size, self.num_frames + 1)
        x = self.sequence_model(x)
        x = self.mlp_head(x)
        return self.softmax(x)

    def create_sliding_window_padded(self, vector, window_size):
        # Initialize output tensor
        output = torch.zeros((self.batch_size, window_size + 1, self.dim), dtype=vector.dtype, device=vector.device)

        for i in range(self.batch_size):
            output[i] = vector[i:i + window_size + 1]

        return output

    def sliding_window(self, tensor, center_count, window_size):
        """
        Args:
        tensor: A tensor of size [sequence_len, feature_dim].
        center_count: Number of central elements to be extracted.
        window_size: Size of the window around each central element.

        Returns:
        A tensor of size [center_count, window_size, feature_dim].
        """
        assert window_size % 2 == 1, "Window size should be odd for symmetric windows around center."
        half_window = window_size // 2

        sequence_len, feature_dim = tensor.shape
        assert sequence_len >= window_size, "Input sequence should be at least as long as the window size."

        # Calculate the starting index of the central elements
        center_start = (sequence_len - center_count) // 2
        center_indices = list(range(center_start, center_start + center_count))

        # For each index in the center_indices list, extract a window and store in a list
        windows = [tensor[i - half_window:i + half_window + 1] for i in center_indices]

        # Stack the windows together to get the result
        result = torch.stack(windows)

        return result


def get_blink_predictor(batch_size: int) -> CNNTransformer:
    model = CNNTransformer(batch_size=batch_size)

    # Get the directory containing the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the parent directory of the script directory
    parent_dir = os.path.dirname(script_dir)

    # Construct the full path to the model checkpoint
    checkpoint_path = os.path.join(parent_dir, "model_blink_completeness_all_data.pt")

    checkpoint = torch.load(checkpoint_path,
                            map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    return model
