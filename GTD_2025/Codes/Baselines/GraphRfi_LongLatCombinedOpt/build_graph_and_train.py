import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch import nn
import numpy as np

from prepare_data import prepare_data
from ndf import GCN, NeuralDecisionForest, GTD100FeatureLayer, Forest

# ====== CONFIG ======
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
NUM_EPOCHS = 2000

def create_models(num_additional_features, num_classes):
    ndf_input_dim = num_additional_features + 1  # GCN error + 15 features

    gcn = GCN(in_channels=2, hidden_channels=100, out_channels=1, activation_fn=F.relu).to(device)
    feature_layer = GTD100FeatureLayer(in_features = 16, dropout_rate=0., shallow=True).to(device)
    forest = Forest(
        n_tree=5,
        tree_depth=4,
        n_in_feature=1024,  # this is correct since GTD100FeatureLayer outputs 1024
        tree_feature_rate=1.0,
        n_class=num_classes
    ).to(device)    
    ndf = NeuralDecisionForest(feature_layer, forest).to(device)

    return gcn, ndf


def train_joint(gcn_model, ndf_model, train_loader, optimizer_gcn, optimizer_ndf, mse_loss_fn):
    

    gcn_model.train()
    ndf_model.train()
    total_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer_gcn.zero_grad()
        optimizer_ndf.zero_grad()

        gcn_out = gcn_model(batch).squeeze()
        weaptype_true = batch.weaptype1.float().to(device)
        gcn_mse = mse_loss_fn(gcn_out, weaptype_true)

        gcn_error = torch.abs(gcn_out - weaptype_true).unsqueeze(1)
        assert gcn_error.ndim == 2 and batch.ndf_features.ndim == 2, f"GCN error shape: {gcn_error.shape}, NDF features shape: {batch.ndf_features.shape}"

        ndf_features = torch.cat([gcn_error, batch.ndf_features.to(device)], dim=1)

        ndf_out = ndf_model(ndf_features)
        ndf_nll = ndf_model.loss(ndf_out, batch.y)

        loss = gcn_mse + ndf_nll
        loss.backward()
        optimizer_gcn.step()
        optimizer_ndf.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate_joint(gcn_model, ndf_model, loader):
    gcn_model.eval()
    ndf_model.eval()
    total_correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        gcn_out = gcn_model(batch).squeeze()
        gcn_error = torch.abs(gcn_out - batch.weaptype1.float().to(device)).unsqueeze(1)
        ndf_features = torch.cat([gcn_error, batch.ndf_features.to(device)], dim=1)
        pred = ndf_model(ndf_features).argmax(dim=1)
        total_correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)

    return total_correct / total


def main(train_path, test_path):
    print("Preparing data...")
    train_list, val_list, class_names = prepare_data(train_path, test_path)
    num_classes = len(class_names)
    num_additional_features = train_list[0].ndf_features.size(0)

    gcn_model, ndf_model = create_models(num_additional_features, num_classes)
    optimizer_gcn = torch.optim.AdamW(gcn_model.parameters(), lr=0.0005)
    optimizer_ndf = torch.optim.AdamW(ndf_model.parameters(), lr=0.005)
    mse_loss_fn = nn.MSELoss()

    train_loader = DataLoader(train_list, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=BATCH_SIZE)

    print("Starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_joint(gcn_model, ndf_model, train_loader, optimizer_gcn, optimizer_ndf, mse_loss_fn)
        val_acc = evaluate_joint(gcn_model, ndf_model, val_loader)
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")


if __name__ == "__main__":
    train_csv = "../../../data/top30groups/LongLatCombined/train1/train100.csv"
    test_csv = "../../../data/top30groups/LongLatCombined/test1/test100.csv"
    main(train_csv, test_csv)
