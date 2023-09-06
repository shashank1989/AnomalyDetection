from sklearn.metrics import precision_recall_fscore_support,accuracy_score
import torch

def Model_Test_Batch_Iteration(x_test, y_test, batch_size, model_name, tokenizer,max_seq_length):
    test_metrics = []

    for batch_start in range(0, len(x_test), batch_size):
        batch_end = batch_start + batch_size
        batch_sequences = x_test[batch_start:batch_end]

        batch_sequence_str = [' '.join(str(num) for num in sequence) for sequence in batch_sequences]

        batch_inputs = tokenizer.batch_encode_plus(
            batch_sequence_str,
            padding='max_length',
            truncation=True,
            max_length=max_seq_length,
            return_tensors='pt'
        )

        input_ids_batch = batch_inputs['input_ids'].to('cuda')
        attention_mask_batch = batch_inputs['attention_mask'].to('cuda')
        y_batch = torch.tensor(y_test[batch_start:batch_end], dtype=torch.long).to('cuda')

        accuracy, precision, recall, f1 = evaluate_batch(input_ids_batch, attention_mask_batch, y_batch, model_name)
        test_metrics.append((accuracy, precision, recall, f1))

    return test_metrics


def evaluate_batch(input_ids_batch, attention_mask_batch, y_batch, model_name):
    model = model_name
    with torch.no_grad():
        test_outputs = model(input_ids_batch, attention_mask_batch)
        _, predicted_labels = torch.max(test_outputs, dim=1)

    predicted_labels = predicted_labels.cpu().numpy()
    accuracy = accuracy_score(y_batch.cpu().numpy(), predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(y_batch.cpu().numpy(), predicted_labels, average='binary')
    return accuracy, precision, recall, f1

def evaluate(model, dataloader):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return accuracy, precision, recall, f1