from torch import nn, Tensor


class MyAwesomeModel(nn.Module):
    """
    A class to define CNN model

    ...

    Attributes
    ----------

    Methods
    -------
    forward(x):
        Computes forward pass of model
    forward_visualize(x):
        Computes forward pass until classification layer
    """

    def __init__(self):
        """Constructs CNN model"""

        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(4 * 7 * 7, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Performs forward pass through model"""

        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

    def forward_visualize(self, x: Tensor) -> Tensor:
        """Computes forward pass until classification layer"""

        x = self.cnn_layers(x)
        return x.flatten()
