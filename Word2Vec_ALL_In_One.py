import jieba
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

jieba.load_userdict("Mydict.txt")

# 準備訓練數據
sentences = [
    "臺灣鐵路已放棄興建的路線",
    "人是一種從動物進化的生物",
    "你是英文系的，可以幫我翻譯一下嗎？"
]

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, center_words, context_words, neg_samples):
        center_embeds = self.center_embeddings(center_words)  # Shape: (batch_size, embedding_dim)
        context_embeds = self.context_embeddings(context_words)  # Shape: (batch_size, embedding_dim)
        neg_embeds = self.context_embeddings(neg_samples)  # Shape: (batch_size, num_neg_samples, embedding_dim)
        
        pos_scores = torch.bmm(context_embeds.view(context_embeds.size(0), 1, context_embeds.size(1)), 
                               center_embeds.view(center_embeds.size(0), center_embeds.size(1), 1)).squeeze()
        
        neg_scores = torch.bmm(neg_embeds.neg(), center_embeds.unsqueeze(2)).squeeze()
        
        return pos_scores, neg_scores

def generate_training_data(corpus, word_to_idx, window_size=2):
    training_data = []
    for sentence in corpus:
        sentence_indices = [word_to_idx[word] for word in sentence]
        for center_pos in range(len(sentence_indices)):
            center_word = sentence_indices[center_pos]
            for w in range(-window_size, window_size + 1):
                context_pos = center_pos + w
                if context_pos < 0 or context_pos >= len(sentence_indices) or center_pos == context_pos:
                    continue
                context_word = sentence_indices[context_pos]
                training_data.append((center_word, context_word))
    return np.array(training_data)

def get_negative_samples(batch_size, num_neg_samples, vocab_size):
    neg_samples = np.random.choice(vocab_size, size=(batch_size, num_neg_samples), replace=True)
    return torch.tensor(neg_samples, dtype=torch.long)

def negative_sampling_loss(pos_scores, neg_scores):
    pos_loss = -F.logsigmoid(pos_scores).mean()
    neg_loss = -F.logsigmoid(-neg_scores).mean()
    return pos_loss + neg_loss

print()
# 將句子分詞
tokenized_sentences = [list(jieba.cut(sentence)) for sentence in sentences]
print("[tokenized_sentences]")
print(tokenized_sentences)
print()

words = [word for sentence in tokenized_sentences for word in sentence]
word_counts = Counter(words)

print("[word_counts]")
print(word_counts)
print()

vocab = sorted(word_counts, key=word_counts.get, reverse=True)

print("[vocab]")
print(vocab)
print()

word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}
vocab_size = len(vocab)
print("[idx_to_word]")
print(idx_to_word)
print()

training_data = generate_training_data(tokenized_sentences, word_to_idx)
print("[training_data]")
print(training_data)
print()


embedding_dim = 100
model = Word2Vec(vocab_size, embedding_dim)

optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 100
num_neg_samples = 5
for epoch in range(num_epochs):
    total_loss = 0
    for center, context in training_data:
        center_tensor = torch.tensor([center], dtype=torch.long)
        context_tensor = torch.tensor([context], dtype=torch.long)
        neg_samples = get_negative_samples(1, num_neg_samples, vocab_size)
        
        optimizer.zero_grad()
        pos_scores, neg_scores = model(center_tensor, context_tensor, neg_samples)
        loss = negative_sampling_loss(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss}")

def get_word_vector(word):
    word_idx = word_to_idx[word]
    word_tensor = torch.tensor([word_idx], dtype=torch.long)
    return model.center_embeddings(word_tensor).detach().numpy()

def find_similar_words(word, top_n=5):
    word_vec = get_word_vector(word)
    similarities = []
    for other_word in vocab:
        if other_word == word:
            continue
        other_vec = get_word_vector(other_word)
        sim = cosine_similarity(word_vec, other_vec)[0][0]
        similarities.append((other_word, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

similar_words = find_similar_words('人')
print(similar_words)