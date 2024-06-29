# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt


def remove_self_loops_and_duplicates(edge_index):
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    edge_index = edge_index.t().contiguous() 
    edge_index = torch.unique(edge_index, dim=0) 
    edge_index = edge_index.t().contiguous() 
    
    return edge_index

class SimpleGNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(SimpleGNN, self).__init__()
        self.conv = GCNConv(num_node_features, 1, bias=False)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index).squeeze(-1)

def generate_data(num_graphs, num_nodes, num_features):
    data_list = []
    for _ in range(num_graphs):
    
        x = torch.randn((num_nodes, num_features))
        random_means = torch.randn(num_nodes).reshape(-1, 1)
        x = x + random_means
        
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_index = remove_self_loops_and_duplicates(edge_index)
        
        y = torch.zeros(num_nodes)
        for i in range(num_nodes):
            neighbors = edge_index[1][edge_index[0] == i]
            y[i] = torch.sum(x[i]) + torch.sum(x[neighbors])
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list

def custom_collate_fn(batch):
    return Batch.from_data_list(batch)

def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_data = generate_data(num_graphs=2000, num_nodes=300, num_features=10)
    test_data = generate_data(num_graphs=200, num_nodes=30, num_features=10)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

    model = SimpleGNN(num_node_features=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
    criterion = torch.nn.MSELoss()

    train_losses = []
    test_losses = []

    for epoch in range(2000):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        scheduler.step()
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
                test_loss += criterion(out, batch.y).item()
        test_losses.append(test_loss / len(test_loader))
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log') 
    plt.show()

    model.eval()
    all_predictions = []
    all_true_values = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            all_predictions.extend(out.cpu().numpy())
            all_true_values.extend(batch.y.cpu().numpy())

    print("\nPredictions vs True Values (first 30 nodes):")
    for i in range(30):
        print(f"Node {i}: Predicted: {all_predictions[i]:.4f}, True: {all_true_values[i]:.4f}")

    mse = np.mean((np.array(all_true_values) - np.array(all_predictions))**2)
    print(f"\nMean Squared Error (MSE): {mse:.4f}")

if __name__ == '__main__':
    main()

