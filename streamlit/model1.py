import torch
import torch.nn as nn

# Define the model class (same as used during training)
class StockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(StockPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU Layer
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate the GRU
        out, _ = self.gru(x, h0)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def load_model(model_path, input_dim = 8, hidden_dim=32, num_layers=2, output_dim=1):
    """
    Load the trained model from a specified path.
    
    Args:
    - model_path (str): Path to the saved model file.
    - input_dim (int): Number of input features.
    - hidden_dim (int, optional): Number of hidden units. Default is 32.
    - num_layers (int, optional): Number of GRU layers. Default is 2.
    - output_dim (int, optional): Number of output units. Default is 1.
    
    Returns:
    - model (nn.Module): The loaded model.
    """
    model = StockPredictor(input_dim, hidden_dim, num_layers, output_dim)
    model.load_state_dict(torch.load(model_path))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    return model
