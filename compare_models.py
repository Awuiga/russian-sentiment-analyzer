"""Оценка двух доверенных локальных моделей на общей отложенной части.

Сохранённые веса должны соответствовать main.py и train_torch.py из проекта.
Сравниваемые модели не переобучаются, checkpoints не меняются.
TF-IDF отдельно восстанавливается только для проверки соответствия train.
"""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split


# Та же архитектура, что в двух пользовательских скриптах; их импорт
# запустил бы обучение или интерактивный ввод.
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.classifier = nn.Linear(embedding_dim, 3)

    def forward(self, input_ids):
        embedding = self.embedding(input_ids)
        mask = (input_ids != 0).unsqueeze(-1)
        vectors = (embedding * mask).sum(dim=1)
        vectors = vectors / mask.sum(dim=1).clamp(min=1)
        return self.classifier(vectors)


LABELS = ['negative', 'neutral', 'positive']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def metrics(actual, predicted):
    return {
        'samples': len(actual),
        'accuracy': accuracy_score(actual, predicted),
        'classification_report': classification_report(
            actual, predicted, labels=LABELS, output_dict=True, zero_division=0
        ),
        'confusion_matrix': confusion_matrix(actual, predicted, labels=LABELS).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/women-clothing-accessories.3-class.balanced.csv')
    parser.add_argument('--sklearn-checkpoint', default='sentiment_model.joblib')
    parser.add_argument('--torch-checkpoint', default='sentiment_torch.pt')
    parser.add_argument('--output', default='reports/comparison.json')
    args = parser.parse_args()
    torch.set_num_threads(2)

    df = pd.read_csv(args.data, sep='\t')
    df['sentiment'] = df['sentiment'].replace({'neautral': 'neutral'})
    counts = df.groupby('review')['sentiment'].nunique()
    df = df[~df['review'].isin(counts[counts > 1].index)].copy()
    df = df.drop_duplicates(subset=['review']).reset_index(drop=True)
    if df['review'].isna().any() or not set(df['sentiment']).issubset(LABELS):
        raise ValueError('Неожиданные пустые тексты или метки: нужны данные исходных экспериментов.')

    positions = np.arange(len(df))
    v1_train, v1_test = train_test_split(
        positions, test_size=.2, random_state=42, stratify=df['sentiment']
    )
    v1_fit, v1_val = train_test_split(
        v1_train, test_size=.2, random_state=42,
        stratify=df.iloc[v1_train]['sentiment']
    )
    v2_train, v2_val = random_split(
        range(len(df)), [.8, .2], generator=torch.Generator().manual_seed(42)
    )
    common = np.intersect1d(v1_test, v2_val.indices)
    if len(common) < 100:
        raise ValueError('Слишком мало общих отложенных примеров.')
    assert not set(common).intersection(v1_fit)
    assert not set(common).intersection(v1_val)
    assert not set(common).intersection(v2_train.indices)

    # Эти файлы исполняют/содержат сериализованные данные: только свои доверенные модели.
    baseline = joblib.load(args.sklearn_checkpoint)
    cp = torch.load(args.torch_checkpoint, map_location='cpu', weights_only=True)
    stoi = cp['stoi']
    expected_stoi = {'<PAD>': 0, '<UNK>': 1}
    words = set()
    for review in df.iloc[v2_train.indices]['review']:
        words.update(review.lower().split())
    expected_stoi.update({word: i for i, word in enumerate(sorted(words), 2)})
    if stoi != expected_stoi or cp['label_to_id'] != {label: i for i, label in enumerate(LABELS)}:
        raise ValueError('Словарь или метки checkpoint не совпали с восстановленным train v2.')
    recreated = TfidfVectorizer(**baseline['vectorizer'].get_params())
    recreated.fit(df.iloc[v1_fit]['review'])
    if (recreated.vocabulary_ != baseline['vectorizer'].vocabulary_
            or not np.allclose(recreated.idf_, baseline['vectorizer'].idf_)):
        raise ValueError('TF-IDF не соответствует восстановленному train v1.')
    embedding_dim = cp['model_state_dict']['embedding.weight'].shape[1]
    model = SentimentClassifier(len(stoi), embedding_dim)
    model.load_state_dict(cp['model_state_dict'], strict=True)
    model.eval()

    def predict_torch(texts):
        result = []
        with torch.no_grad():
            for start in range(0, len(texts), 64):
                encoded = [[stoi.get(w, 1) for w in text.lower().split()] or [1]
                           for text in texts[start:start + 64]]
                max_len = max(map(len, encoded))
                x = torch.tensor([row + [0] * (max_len - len(row)) for row in encoded])
                ids = model(x).argmax(dim=1).tolist()
                result.extend(LABELS[i] for i in ids)
        return result

    common_texts = df.iloc[common]['review'].tolist()
    actual = df.iloc[common]['sentiment'].tolist()
    p1 = baseline['model'].predict(baseline['vectorizer'].transform(common_texts)).tolist()
    p2 = predict_torch(common_texts)
    native_val = df.iloc[v2_val.indices]
    both_correct = sum(a == x == y for a, x, y in zip(actual, p1, p2))
    only_v1 = sum(a == x and a != y for a, x, y in zip(actual, p1, p2))
    only_v2 = sum(a != x and a == y for a, x, y in zip(actual, p1, p2))
    report = {
        'evaluation_date': datetime.now(timezone(timedelta(hours=3))).date().isoformat(),
        'method': 'intersection of v1 test and v2 validation; reconstructed splits, seed 42',
        'limitations': [
            'Assumes checkpoints were trained by the supplied scripts on this corpus.',
            'Checkpoints do not store training indices or dataset hashes.',
            'Exact v2 training vocabulary and v1 TF-IDF vocabulary/IDF matched.',
            'Training sizes and tokenization differ; this is a comparison of saved systems.',
            'V2 validation may already have been inspected during development.',
            'No claim of an untouched final benchmark or causal architecture improvement.',
        ],
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'pandas': pd.__version__, 'sklearn': sklearn.__version__},
        'hashes': {'dataset': digest(args.data), 'v1_checkpoint': digest(args.sklearn_checkpoint),
                   'v2_checkpoint': digest(args.torch_checkpoint),
                   'common_positions_int64_le': hashlib.sha256(common.astype('<i8').tobytes()).hexdigest()},
        'split_sizes': {'cleaned': len(df), 'v1_train': len(v1_fit), 'v1_val': len(v1_val),
                        'v1_test': len(v1_test), 'v2_train': len(v2_train),
                        'v2_val': len(v2_val), 'common': len(common)},
        'labels': LABELS, 'v2_vocabulary_size': len(stoi),
        'v2_parameters': sum(p.numel() for p in model.parameters()),
        'common_v1': metrics(actual, p1), 'common_v2': metrics(actual, p2),
        'v2_native_validation': metrics(native_val['sentiment'].tolist(),
                                        predict_torch(native_val['review'].tolist())),
        'paired_counts': {'both_correct': both_correct, 'only_v1_correct': only_v1,
                          'only_v2_correct': only_v2,
                          'both_wrong': len(common) - both_correct - only_v1 - only_v2},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for key in ['split_sizes', 'v2_vocabulary_size', 'v2_parameters', 'paired_counts']:
        print(key, report[key])
    for key in ['common_v1', 'common_v2', 'v2_native_validation']:
        print(key, 'accuracy', report[key]['accuracy'],
              'macro F1', report[key]['classification_report']['macro avg']['f1-score'])


if __name__ == '__main__':
    main()
