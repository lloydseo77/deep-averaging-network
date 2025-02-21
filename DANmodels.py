# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, read_word_embeddings, WordEmbeddings
from torch.utils.data import Dataset

# Dataset class for handling sentiment analysis data
class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, embeddings_file, max_length=100, train=True):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)

        # Read word embeddings
        self.word_embeddings = read_word_embeddings(embeddings_file)
        
        # Extract label from the examples
        self.labels = [ex.label for ex in self.examples]

        # Convert sentences to padded lists of word indices
        self.max_length = max_length
        self.word_indices = [self.convert_to_indices(ex.words) for ex in self.examples]
        
    def convert_to_indices(self, words):
        indices = [self.word_embeddings.word_indexer.index_of(word) for word in words]

        # Handle unknown words
        indices = [index if index != -1 else self.word_embeddings.word_indexer.index_of("UNK") for index in indices]

        # Padding
        if len(indices) < self.max_length:
            indices += [self.word_embeddings.word_indexer.index_of("PAD")] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        
        return indices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the word indices and label for the given index
        word_indices = self.word_indices[idx]
        label = self.labels[idx]
        
        return torch.tensor(word_indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# Two-layer fully connected neural network
class NN2DAN(nn.Module):
    def __init__(self, embeddings: WordEmbeddings, hidden_size):
        super().__init__()

        # Embedding layer
        self.embedding = embeddings.get_initialized_embedding_layer(self)

        # Fully-connected layers
        self.fc1 = nn.Linear(embeddings.get_embedding_length(), hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

        # LogSoftmax -- Final output
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Ensure input is of the correct type
        if x.dtype != torch.long:
            x = x.long()  # Convert to long if it's not

        embedded = self.embedding(x)

        # Calculate average of the embeddings
        mask = (x != 0).float()  # Create a mask to ignore PAD tokens
        summed = torch.sum(embedded * mask.unsqueeze(-1), dim=1)  # Sum embeddings over the sentence length
        lengths = torch.sum(mask, dim=1, keepdim=True)  # Get the lengths of non-pad tokens
        averaged = summed / lengths  # Compute the mean embeddings for each sentence

        # Pass averaged embedding through network
        x = F.relu(self.fc1(averaged))
        x = self.fc2(x)

        # Return log prob
        x = self.log_softmax(x)
        return x
    
# Three-layer fully connected neural network
class NN3DAN(nn.Module):
    def __init__(self, embeddings: WordEmbeddings, hidden_size):
        super().__init__()

        # Embedding layer
        self.embedding = embeddings.get_initialized_embedding_layer(self)

        # Fully-connected layers
        self.fc1 = nn.Linear(embeddings.get_embedding_length(), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)

        # LogSoftmax -- Final output
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Ensure input is of the correct type
        if x.dtype != torch.long:
            x = x.long()  # Convert to long if it's not

        embedded = self.embedding(x)

        # Calculate average of the embeddings
        mask = (x != 0).float()  # Create a mask to ignore PAD tokens
        summed = torch.sum(embedded * mask.unsqueeze(-1), dim=1)  # Sum embeddings over the sentence length
        lengths = torch.sum(mask, dim=1, keepdim=True)  # Get the lengths of non-pad tokens
        averaged = summed / lengths  # Compute the mean embeddings for each sentence

        # Pass averaged embedding through network
        x = F.relu(self.fc1(averaged))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Return log prob
        x = self.log_softmax(x)
        return x
