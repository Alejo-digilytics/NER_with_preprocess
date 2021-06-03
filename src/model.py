import torch.nn as nn
import src.config as config
from src.train_val_loss import loss_function
from pytorch_pretrained_bert import BertModel


class BERT_NER(nn.Module):
    """
    This class is the DL model. It can be tuned to use one or two different extra layers with different dropouts per
    classification: tag, or ner
    """
    def __init__(self, num_tag,
                 base_model="bert-base-uncased",
                 tag_dropout=0.3,
                 tag_dropout_2=0.3,
                 architecture="simple",
                 middle_layer=100,
                 freezing=False):
        super(BERT_NER, self).__init__()

        # base model and architecture
        self.base_model = base_model
        self.architecture = architecture

        if base_model == "bert-base-uncased":
            self.model = BertModel.from_pretrained(config.BERT_UNCASED_PATH)
        if base_model == "finbert-uncased":
            self.model = BertModel.from_pretrained(config.FINBERT_UNCASED)
        if base_model == "mortbert-uncased":
            self.model = BertModel.from_pretrained(config.MORTBERT_UNCASED)

        if freezing:
            for param in self.model.parameters():
                param.requires_grad = False

        # NER parameters
        self.num_tag = num_tag

        # First dropout
        self.bert_drop_tag_1 = nn.Dropout(tag_dropout)

        # Architecture
        if self.architecture == "simple":
            # 768 (BERT) composed with a linear function
            self.out_tag = nn.Linear(self.model.config.hidden_size, self.num_tag)

        if self.architecture == "complex":
            # 768 (BERT) composed with a linear function
            self.tag_mid = nn.Linear(self.model.config.hidden_size, middle_layer)
            # self.ac = nn.GELU()
            self.bert_drop_tag_2 = nn.Dropout(tag_dropout_2)
            self.out_tag = nn.Linear(middle_layer, self.num_tag)

    def forward(self, ids, mask, tokens_type_ids, target_tag):
        """
        This method if the extra fine tuning NN for tags
        """
        # Since this model is for NER we need to take the sequence output
        # We don't want to get a value as output but a sequence of outputs, one per token
        # BERT sequence output is the first output. Here o1
        o1, _ = self.model(input_ids=ids,
                           token_type_ids=tokens_type_ids,
                           attention_mask=mask,
                           output_all_encoded_layers=False
                           )

        # Simple architecture
        if self.architecture == "simple":
            output_tag = self.bert_drop_tag_1(o1)

        # Complex architecture
        if self.architecture == "complex":
            # Add dropout 1
            output_tag1 = self.bert_drop_tag_1(o1)
            # Add middle layer
            output_tag_2 = self.tag_mid(output_tag1)
            # output_tag_2 = self.ac(output_tag_2)
            # Add second dropout
            output_tag = self.bert_drop_tag_2(output_tag_2)

        # We add the linear outputs
        tag = self.out_tag(output_tag)
        return tag, loss_function(tag, target_tag, mask, self.num_tag)
