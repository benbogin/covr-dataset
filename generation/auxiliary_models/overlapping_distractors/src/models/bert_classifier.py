from typing import Dict, Optional

import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from transformers import AutoModelForSequenceClassification


@Model.register("term_pretrained_for_classification")
class pretrainedForClassification(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 tokens_namespace: str = "tokens",
                 regularizer: Optional[RegularizerApplicator] = None,) -> None:
        super().__init__(vocab, regularizer)

        self.pretrained_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        self._tokens_namespace = tokens_namespace

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: Dict = None,
                ) -> Dict[str, torch.Tensor]:
        input_ids = tokens[self._tokens_namespace]['token_ids']
        token_type_ids = tokens[self._tokens_namespace]['type_ids']
        input_mask = tokens[self._tokens_namespace]['mask']

        model_output = self.pretrained_model(input_ids=input_ids,
                                          token_type_ids=token_type_ids,
                                          attention_mask=input_mask)

        # apply classification layer
        logits = model_output.logits

        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            output_dict["metadata"] = metadata
            output_dict["correct"] = probs.argmax(dim=-1) == label
            self._accuracy(logits, label)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {'accuracy': self._accuracy.get_metric(reset)}
        return metrics