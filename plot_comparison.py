"""График по reports/comparison.json: одинаковая отложенная часть и шкала."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
report = json.loads((root / 'reports/comparison.json').read_text(encoding='utf-8'))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
maximum = max(v for key in ['common_v1', 'common_v2']
              for row in report[key]['confusion_matrix'] for v in row)
for ax, key, name in zip(axes, ['common_v1', 'common_v2'],
                         ['v1 · TF-IDF + Logistic Regression', 'v2 · PyTorch · среднее Embedding']):
    result = report[key]
    matrix = result['confusion_matrix']
    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=maximum)
    ax.set(xticks=range(3), yticks=range(3), xticklabels=report['labels'],
           yticklabels=report['labels'], xlabel='Предсказанный класс',
           ylabel='Истинный класс',
           title=f"{name}\nAccuracy {result['accuracy']:.2%} · macro-F1 {result['classification_report']['macro avg']['f1-score']:.3f}")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i][j]), ha='center', va='center',
                    color='white' if matrix[i][j] > maximum / 2 else '#0f172a')
fig.suptitle(f"Общие отложенные отзывы: {report['split_sizes']['common']} · оценка {report['evaluation_date']}")
fig.colorbar(im, ax=axes, shrink=.75, label='Количество отзывов')
fig.savefig(root / 'reports/comparison.png', dpi=160)
plt.close(fig)
