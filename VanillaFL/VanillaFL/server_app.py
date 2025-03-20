"""VanillaFL: A Flower / PyTorch app."""
import flwr as fl
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
import torch
from torch import nn
from VanillaFL.task import Net, get_weights

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(64 * (79 - 4), 128)        # 42 -> input dimension(NSLKDD) 79 -> INSDN
        self.fc2 = nn.Linear(128, 2)  # Binary Classifier

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    num_clients = context.run_config["num-clients"]
    model = Net()
    model.load_state_dict(torch.load("/home/shikhar/Desktop/FLResearch/Code/Initial_Global_Model.pth")) # Path of the initial global model
    # Get model params
    ndarrays = get_weights(model)
    parameters = ndarrays_to_parameters(ndarrays)

    # Federated Averaging Strategy
    strategy = fl.server.strategy.FedAvg(
        # fraction_fit=fraction_fit,        # Uncomment for sampling out of total
        # fraction_evaluate=1.0,
        # proximal_mu= 0.1 ,                # Use in case of FedProx
        min_available_clients=num_clients,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
