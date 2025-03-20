"""VanillaFL: A Flower / PyTorch app."""

from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.optim as optim
from torch.utils.data import Subset
from utils.Poisoning import flip_labels

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

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


def load_data(partition_id: int, num_partitions: int):
    # Load training dataset
    train_df = pd.read_csv('/home/shikhar/Desktop/FLResearch/Code/Dataset/INSDN/INSDN_Dataset_Train_Client.csv', low_memory=False)
    train_df.fillna(0, inplace=True)  # Fill missing values with zero

    # Split features and labels
    features_train = train_df.iloc[:, :-1].values
    labels_train = train_df['Label'].values  # (0,1) Binary Labels

    # Normalize features
    scaler = MinMaxScaler()
    features_train = scaler.fit_transform(features_train)

    # Create dataset and partition it
    train_dataset = CustomDataset(features_train, labels_train)
    dataset_size = len(train_dataset)
    indices = np.random.permutation(dataset_size)  # Shuffle indices
    split_indices = np.array_split(indices, num_partitions)  # Split into `num_partitions`

    # Subset for each client
    client_datasets = {client_id: Subset(train_dataset, idx) for client_id, idx in enumerate(split_indices)}
    client_train_dataset = client_datasets[partition_id]

    # Convert back to df
    data_list = []
    for features, label in client_train_dataset:
        features = features.numpy().tolist()
        data_list.append(features + [label.item()])

    column_names = [f'feature_{i}' for i in range(len(features))] + ['Label']
    df = pd.DataFrame(data_list, columns=column_names)



    """ This is the beginning of poisoning script ..... Uncomment the below code to poison client database."""
    # poisoned_client = 0
    # flip_percentage = 0.99
    # if partition_id == poisoned_client:
    #     df_flipped = flip_labels(df, label_column='Label', flip_percentage=flip_percentage)
    #     df = df_flipped
    #     print(f"The poisoned client is : {partition_id}")
    """ END OF POISONING SCRIPT"""




    print(f"The size of the train dataset of client {partition_id} is {len(df)}")

    # Convert back to PyTorch Dataset before using DataLoader
    train_dataset_filtered = CustomDataset(df.iloc[:, :-1].values, df['Label'].values)
    train_loader = DataLoader(train_dataset_filtered, batch_size=32, shuffle=True)

    # Load and process test dataset
    test_df = pd.read_csv('/home/shikhar/Desktop/FLResearch/Code/Dataset/INSDN/INSDN_Dataset_Test.csv', low_memory=False)
    test_df.fillna(0, inplace=True)
    features_test = test_df.iloc[:, :-1].values
    labels_test = test_df['Label'].values

    # Normalize test features
    features_test = scaler.transform(features_test)

    # Convert to PyTorch Dataset and DataLoader
    test_dataset = CustomDataset(features_test, labels_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    total_loss_over_epochs = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for data, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(data)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        avg_train_loss = running_loss / len(trainloader)  # Average loss for current epoch

        total_loss_over_epochs += avg_train_loss  # Sum of average loss across epochs

        print(f'Epoch {epoch + 1}, Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

    # Final average loss over all epochs
    overall_avg_train_loss = total_loss_over_epochs / epochs
    print(f'Overall Average Train Loss: {overall_avg_train_loss:.4f}')

    return overall_avg_train_loss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()  # Set the model to evaluation mode
    correct, total = 0, 0
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, labels in testloader:
            outputs = net(data)
            loss = loss_fn(outputs, labels)  # Compute loss
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)  # Get predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(testloader)  # Calculate average loss
    accuracy = 100 * correct / total  # Compute accuracy
    print(f"Test accuracy : {accuracy} and loss : {avg_loss}")
    return avg_loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)