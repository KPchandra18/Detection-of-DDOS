import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import dgl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Define the GraphDDoS model
class GraphDDoS(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphDDoS, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load your dataset
data = pd.read_excel("C:/Users/kpcha/newenv/OneDrive/Desktop/New Microsoft Excel Worksheet.xlsx")        
# Define your dataset features and labels
# Assuming the last column ('target') indicates 'normal' or 'DDoS'
non_numeric_columns = ['src_ip', 'dst_ip', 'timestamp']
numeric_features = data.drop(columns=non_numeric_columns)
features = numeric_features.values
labels = data['target'].values

# Encode labels ('normal' as 0, 'DDoS' as 1)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=42)

# Standardize feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to DGL graphs (not considering edge features for now)
def create_dgl_graphs(data):
    graphs = []
    for i in range(len(data)):
        # Create DGL graph and add nodes
        g = dgl.DGLGraph()
        num_nodes = len(data[i])
        g.add_nodes(num_nodes)

        # Create edges (assuming a fully connected graph)
        src_nodes, dst_nodes = np.meshgrid(range(num_nodes), range(num_nodes))
        src_nodes, dst_nodes = src_nodes.flatten(), dst_nodes.flatten()
        g.add_edges(src_nodes, dst_nodes)

        graphs.append(g)

    return graphs

train_graphs = create_dgl_graphs(X_train)
test_graphs = create_dgl_graphs(X_test)

# Define a custom collate function for the DataLoader
def custom_collate(batch):
    graphs, features, labels = zip(*batch)
    bg = dgl.batch(graphs)
    features = np.array(features)
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    return bg, features, labels

# Create DataLoader instances for training and testing with the custom collate function
batch_size = 32
train_dataset = list(zip(train_graphs, X_train, y_train))
test_dataset = list(zip(test_graphs, X_test, y_test))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate)

# Initialize and train the GraphDDoS model
input_dim = len(X_train[0])
hidden_dim = 64
output_dim = 2
learning_rate = 0.001
num_iters = 10

model = GraphDDoS(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(num_iters):
    model.train()
    for batch_data in train_dataloader:
        optimizer.zero_grad()
        graphs, features, labels = batch_data
        output = model(features)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model on the test set
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_data in test_dataloader:
        graphs, features, labels = batch_data
        output = model(features)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')
