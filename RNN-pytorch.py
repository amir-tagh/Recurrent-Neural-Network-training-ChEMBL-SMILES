import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm, trange
import argparse

# Custom Dataset for SMILES strings
class SMILESDataset(Dataset):
    def __init__(self, smiles_list, char_to_idx, max_len):
        self.smiles_list = smiles_list
        self.char_to_idx = char_to_idx
        self.max_len = max_len
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        encoded = self.encode_smiles(smiles)
        return torch.tensor(encoded, dtype=torch.long)
    
    def encode_smiles(self, smiles):
        encoded = [self.char_to_idx[char] for char in smiles]
        if len(encoded) < self.max_len:
            encoded += [self.char_to_idx['<PAD>']] * (self.max_len - len(encoded))
        return encoded

def load_smiles_data(filepath, max_len):
    df = pd.read_csv(filepath, sep='\t')
    smiles_list = df.iloc[:, 0].tolist()

    # Create character to index mapping
    chars = sorted(set(''.join(smiles_list)))
    char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}
    char_to_idx['<PAD>'] = 0  # Padding index
    
    dataset = SMILESDataset(smiles_list, char_to_idx, max_len)
    return dataset, char_to_idx

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        if hidden is None:
            lstm_out, hidden = self.lstm(embedded)
        else:
            lstm_out, hidden = self.lstm(embedded, hidden)
        output = self.fc(lstm_out)
        return output, hidden

def train(model, dataloader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.train()
    for epoch in trange(num_epochs, desc="Epochs"):
        total_loss = 0
        for batch in tqdm(dataloader, desc="Batches", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs, _ = model(batch[:, :-1])
            loss = criterion(outputs.view(-1, outputs.shape[-1]), batch[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')

def evaluate(model, smiles, char_to_idx, max_len, device='cuda'):
    model.eval()
    encoded = torch.tensor([char_to_idx[char] for char in smiles] + [char_to_idx['<PAD>']] * (max_len - len(smiles)), dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output, _ = model(encoded[:, :-1])
    _, predicted = torch.max(output, 2)
    decoded = ''.join([list(char_to_idx.keys())[list(char_to_idx.values()).index(idx)] for idx in predicted.squeeze().cpu().numpy()])
    return decoded

# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an LSTM on SMILES data')
    parser.add_argument('--smiles_file', type=str, required=True, help='Path to the tab-separated SMILES file')
    parser.add_argument('--max_len', type=int, default=100, help='Maximum length of SMILES strings')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of embeddings')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Dimension of hidden layers')
    parser.add_argument('--output_dim', type=int, default=128, help='Output dimension of the model')
    args = parser.parse_args()

    dataset, char_to_idx = load_smiles_data(args.smiles_file, args.max_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    vocab_size = len(char_to_idx)
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    output_dim = vocab_size

    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parallelize the model across multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)

    train(model, dataloader, criterion, optimizer, num_epochs=args.epochs, device=device)

    # Example evaluation
    smiles = "CCO"
    decoded_smiles = evaluate(model, smiles, char_to_idx, args.max_len, device=device)
    print(f'Original: {smiles}, Decoded: {decoded_smiles}')

