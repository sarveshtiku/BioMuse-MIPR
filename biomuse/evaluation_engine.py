from sklearn.metrics import f1_score

def evaluate_retrieval(predicted, ground_truth):
    overlap = len(set(predicted) & set(ground_truth))
    recall = overlap / len(ground_truth) if ground_truth else 0
    precision = overlap / len(predicted) if predicted else 0
    return {'precision': precision, 'recall': recall}

def evaluate_tag_prediction(predicted_tags, gold_tags):
    y_true = [1 if tag in gold_tags else 0 for tag in predicted_tags]
    y_pred = [1] * len(predicted_tags)
    if len(set(gold_tags + predicted_tags)) == 0:
        f1 = 1.0
    else:
        f1 = f1_score(y_true, y_pred, average='macro')
    return {'f1': f1}

# For summarization: use rouge-score package externally (add later)