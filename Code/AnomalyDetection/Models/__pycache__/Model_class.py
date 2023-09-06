import torch.nn as nn

#BaseBert
class ClassificationBERTModel(nn.Module):
    def __init__(self, bert_model, num_labels=2):
        super(ClassificationBERTModel, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']
        logits = self.classifier(pooled_output)
        return logits


# Create a simple binary classification model using BERT embeddings
class ClassificationTunedBERTModel(nn.Module):
    def __init__(self, bert_model, num_labels=2):
        super(ClassificationTunedBERTModel, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']
        logits = self.classifier(pooled_output)
        return logits

class ClassificationDistilBERTModel(nn.Module):
    def __init__(self, distilbert_model, num_labels=2):
        super(ClassificationDistilBERTModel, self).__init__()
        self.distilbert = distilbert_model
        self.classifier = nn.Sequential(
            nn.Linear(self.distilbert.config.dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs['last_hidden_state'][:, 0, :]
        logits = self.classifier(pooled_output)
        return logits

class ClassificationRoBERTaModel(nn.Module):
    def __init__(self, roberta_model, num_labels=2):
        super(ClassificationRoBERTaModel, self).__init__()
        self.roberta = roberta_model
        self.classifier = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']
        logits = self.classifier(pooled_output)
        return logits

class ClassificationAlbertModel(nn.Module):
    def __init__(self, albert_model, num_labels=2):
        super(ClassificationAlbertModel, self).__init__()
        self.albert = albert_model
        self.classifier = nn.Sequential(
            nn.Linear(self.albert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.albert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']
        logits = self.classifier(pooled_output)
        return logits

from transformers import XLNetModel, XLNetTokenizer
class ClassificationXLNetModel(nn.Module):
    def __init__(self, xlnet_model, num_labels=2):
        super(ClassificationXLNetModel, self).__init__()
        self.xlnet = xlnet_model
        self.classifier = nn.Sequential(
            nn.Linear(self.xlnet.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.xlnet(input_ids, attention_mask=attention_mask)
        pooled_output = outputs['last_hidden_state'][:, 0]
        logits = self.classifier(pooled_output)
        return logits