import torch
import torch.nn as nn

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

check_point = torch.load("sentiment_torch.pt",
           map_location = 'cpu',
           weights_only = True
           )

stoi = check_point['stoi']
model = SentimentClassifier(len(stoi), 32)
model.load_state_dict(check_point["model_state_dict"])
model.eval()

review = input()
words = review.lower().split()
encoded = []
for word in words:
    encoded.append(stoi.get(word, 1))

if not encoded:
    encoded.append(1)

input_ids = torch.tensor([encoded], dtype = torch.long)

with torch.no_grad():
    logits = model(input_ids)
    predictions = logits.argmax(dim = 1)

label_to_id = check_point['label_to_id']
reverse = {id_to_label: key for key, id_to_label in label_to_id.items()}
prediction_id = predictions.item()
print(reverse[prediction_id])
