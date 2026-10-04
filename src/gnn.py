import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HeteroConv, SAGEConv
from sklearn.metrics import classification_report, f1_score


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        self.head = torch.nn.Linear(hidden_channels, out_channels)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index).relu()
        
        return x, self.head(x)


    @property
    def device(self):
        return next(self.parameters()).device


def train_model(model, data, epochs=100, lr=0.005, train_mask=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train_mask = data.pseudo_y != -1 if train_mask is None else (train_mask.to(device) & (data.pseudo_y != -1))
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        _, out = model(data.x, data.edge_index)
        
        loss = criterion(out[train_mask], data.pseudo_y[train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f}')


@torch.no_grad()
def evaluate_model(model, data, output_report_dict=False, test_mask=None):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    test_mask = torch.ones(data.num_nodes, dtype=torch.bool) if test_mask is None else test_mask.to(device)

    embs, out = model(data.x, data.edge_index)
    preds = out.argmax(dim=1)
    
    y_true = data.y.cpu().numpy()
    y_pred = preds.cpu().numpy()

    report = classification_report(y_true[test_mask.cpu().numpy()], 
                                   y_pred[test_mask.cpu().numpy()], 
                                   zero_division=0, target_names=data.id2label.values(), 
                                   output_dict=output_report_dict)
    
    f1_macro = f1_score(y_true[test_mask.cpu().numpy()], y_pred[test_mask.cpu().numpy()], average='macro')

    return report, f1_macro, torch.tensor(y_pred)[test_mask.cpu()], embs


import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = HeteroConv({
            ('text', 'similar', 'text'): SAGEConv(-1, hidden_channels),
            ('text', 'matches', 'description'): SAGEConv(-1, hidden_channels),
            ('description', 'rev_matches', 'text'): SAGEConv(-1, hidden_channels),
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('text', 'similar', 'text'): SAGEConv(hidden_channels, hidden_channels),
            ('text', 'matches', 'description'): SAGEConv(hidden_channels, hidden_channels),
            ('description', 'rev_matches', 'text'): SAGEConv(hidden_channels, hidden_channels),
        }, aggr='sum')

        self.head = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = {k: F.dropout(v, p=0.4, training=self.training)
                  for k, v in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        # Only classify text nodes
        return x_dict['text'], self.head(x_dict['text'])

    @property
    def device(self):
        return next(self.parameters()).device
    

def train_hetero_model(model, data, epochs=100, lr=0.005, train_mask=None, llm_labels=None, alpha=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    text_data = data['text']

    if train_mask is None:
        train_mask = text_data.pseudo_y != -1
    else:
        train_mask = train_mask.to(device) & (text_data.pseudo_y != -1)

    # Process LLM label mask
    if llm_labels is not None:
        llm_labels = llm_labels.to(device).bool()

        llm_mask = train_mask & llm_labels
        other_mask = train_mask & (~llm_labels)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        _, out = model(data.x_dict, data.edge_index_dict)

        if llm_labels is None:
            loss = criterion(out[train_mask], text_data.pseudo_y[train_mask])
        else:
            loss = 0.0

            if llm_mask.sum() > 0:
                loss_knn = criterion(out[llm_mask], text_data.pseudo_y[llm_mask])
                loss = 2*(loss + alpha * loss_knn)

            if other_mask.sum() > 0:
                loss_other = criterion(out[other_mask], text_data.pseudo_y[other_mask])
                loss = 2*(loss + (1 - alpha) * loss_other)

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            if llm_labels is None:
                print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f}')
            else:
                print(
                    f'Epoch {epoch:03d} | Loss: {loss.item():.4f} | '
                    f'LLM: {llm_mask.sum().item()} | '
                    f'Other: {other_mask.sum().item()}'
                )


from sklearn.metrics import classification_report, f1_score


@torch.no_grad()
def evaluate_hetero_model(model, data, output_report_dict=False, test_mask=None):
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    text_data = data['text']

    if test_mask is None:
        test_mask = torch.ones(text_data.num_nodes, dtype=torch.bool, device=device)
    else:
        test_mask = test_mask.to(device)

    embs, out = model(data.x_dict, data.edge_index_dict)
    preds = out.argmax(dim=1)

    y_true = text_data.y.cpu().numpy()
    y_pred = preds.cpu().numpy()

    mask_np = test_mask.cpu().numpy()

    report = classification_report(
        y_true[mask_np],
        y_pred[mask_np],
        zero_division=0,
        target_names=list(text_data.id2label.values()),
        output_dict=output_report_dict
    )

    f1_macro = f1_score(
        y_true[mask_np],
        y_pred[mask_np],
        average='macro'
    )

    return report, f1_macro, torch.tensor(y_pred)[test_mask.cpu()], embs