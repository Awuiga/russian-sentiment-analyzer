
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# 1. Загрузка и очистка данных

df = pd.read_csv(
    "data/women-clothing-accessories.3-class.balanced.csv",
    sep="\t"
)

df["sentiment"] = df["sentiment"].replace({
    "neautral": "neutral"
})

label_counts = df.groupby("review")["sentiment"].nunique()
conflicts = label_counts[label_counts > 1].index

df = df[~df["review"].isin(conflicts)].copy()
df = df.drop_duplicates(subset=["review"]).copy()


# 2. Dataset и разделение данных

label_to_id = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}


class ReviewsDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        review = self.df["review"].iloc[idx]
        sentiment = self.df["sentiment"].iloc[idx]
        sentiment_id = label_to_id[sentiment]

        return review, sentiment_id


dataset = ReviewsDataset(df)

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [0.8, 0.2],
    generator=generator
)


# 3. Словарь по Train

all_train_words = set()

for review, sentiment_id in train_dataset:
    words = review.lower().split()
    all_train_words.update(words)

stoi = {
    "<PAD>": 0,
    "<UNK>": 1
}

for i, word in enumerate(sorted(all_train_words), start=2):
    stoi[word] = i


# 4. Кодирование и Padding батчей

def collate_reviews(batch):
    encoded_reviews = []
    targets = []

    for review, sentiment_id in batch:
        words = review.lower().split()
        encoded = []

        for word in words:
            encoded.append(stoi.get(word, 1))

        if not encoded:
            encoded.append(1)

        encoded_reviews.append(encoded)
        targets.append(sentiment_id)

    max_length = max(len(encoded) for encoded in encoded_reviews)

    for encoded in encoded_reviews:
        other_length = max_length - len(encoded)
        encoded.extend([0] * other_length)

    input_ids = torch.tensor(encoded_reviews, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return input_ids, targets


train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_reviews
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_reviews
)


# 5. Нейросеть

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.classifier = nn.Linear(embedding_dim, 3)

    def forward(self, input_ids):
        embedding = self.embedding(input_ids)

        mask = (input_ids != 0).unsqueeze(-1)

        vectors = (embedding * mask).sum(dim=1)
        vectors = vectors / mask.sum(dim=1).clamp(min=1)

        logits = self.classifier(vectors)

        return logits


# 6. Подготовка обучения

model = SentimentClassifier(len(stoi), 32)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

# 7. Обучение
model.train()
total_loss = 0.0
for step ,(input_ids, targets) in enumerate(train_dataloader):
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = criterion(logits, targets)
    loss.backward()
    total_loss += loss.item()
    optimizer.step()
    if (step + 1) % 1000 == 0:
        print('Step', step + 1,
              'Loss', loss.item())

avg_loss = total_loss / len(train_dataloader)

# 8. Валидация

model.eval()
val_total_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for input_ids, targets in val_dataloader:
        logits = model(input_ids)
        predictions = logits.argmax(dim = 1)
        loss = criterion(logits, targets)
        correct += sum(predictions == targets).item()
        total += len(targets)
        val_total_loss += loss.item()
val_avg_loss = val_total_loss / len(val_dataloader)
accuracy = (correct / total) * 100
print("Train Loss:", avg_loss)
print("Validation Loss:", val_avg_loss)
print("Accuracy:", accuracy)

check_point = {
    'model_state_dict': model.state_dict(),
    'stoi': stoi,
    'label_to_id': label_to_id
}
torch.save(check_point, 'sentiment_torch.pt')
