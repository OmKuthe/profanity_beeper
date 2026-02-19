import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer

# Dummy dataset
texts = [
    "you are stupid",
    "have a nice day",
    "i hate you",
    "good morning",
    "you idiot"
]

labels = torch.tensor([1, 0, 1, 0, 1], dtype=torch.float32)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()
X = torch.tensor(X, dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, 16, batch_first=True)
        self.fc = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return self.sigmoid(out)

model = LSTMModel(X.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Quick training
for _ in range(20):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs.squeeze(), labels)
    loss.backward()
    optimizer.step()

def detect(words):
    flagged = []

    for w in words:
        vec = vectorizer.transform([w["word"]]).toarray()
        vec = torch.tensor(vec, dtype=torch.float32)

        with torch.no_grad():
            prediction = model(vec)

        if prediction.item() > 0.5:
            flagged.append(w)

    return flagged
