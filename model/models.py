import torch
import torch.nn as nn


######################## MODELS ########################


class GenomicExpressionNet2(nn.Module):
    """
    A fully connected neural network for multi-task regression
    on GLOBAL Signature.

    Architecture
    ------------
    - Input Layer: Linear(3200 → 2048) + ReLU + Dropout(dropouts[0])
    - Hidden Layer 1: Linear(2048 → 1024) + ReLU + Dropout(dropouts[1])
    - Hidden Layer 2: Linear(1024 → 512) + ReLU + Dropout(dropouts[1])
    - Output Layer: Linear(512 → 1035) + ReLU

    Parameters
    ----------
    dropouts : list of float, length 2
    """

    def __init__(self, dropouts: list[float] = [0.3, 0.2]) -> None:
        super().__init__()

        if not isinstance(dropouts, list) or len(dropouts) != 2:
            raise ValueError("`dropouts` must be a list of 2 float values.")

        for i, d in enumerate(dropouts):
            if not (0.0 <= d <= 1.0):
                raise ValueError(f"Dropout at index {i} must be between 0.0 and 1.0")
        
        self.input1 =  nn.Sequential(
            nn.Linear(3200, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropouts[0]),
        )
        self.input = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropouts[1]),
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropouts[1]),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 1035),
        )

      

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 3200)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1035)
        """
        x = self.input1(x)
        x = self.input(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


