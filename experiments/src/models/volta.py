from collections import defaultdict
from typing import Dict

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.training.metrics import Average
from overrides import overrides

from .volta_src.config import BertConfig
from .volta_src.encoders import BertForVLTasks
from .volta_src.embeddings import BertLayerNorm
from .volta_src.encoders import GeLU
from .extras import convert_sents_to_features, BertLayer

from transformers import BertTokenizer


@Model.register("volta")
class VisualBertEncoder(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 max_seq_length: int,
                 num_images: int,
                 pretrained_model_path: str,
                 pretrained_model_config: str,
                 classifier_transformer_layers: int = 0,
                 classifier_transformer_layers_cls: bool = False
                 ) -> None:
        super().__init__(vocab=vocab)

        # load volta's config
        config = BertConfig.from_json_file(pretrained_model_config)

        self.max_seq_length = max_seq_length
        self.encoder = BertForVLTasks.from_pretrained(pretrained_model_path,
                                                      config=config, num_labels=self.vocab.get_vocab_size("labels"))

        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        self.num_images = max(num_images, 1)
        self.hid_dim = hid_dim = config.pooler_size
        self._num_vis_position_features = config.num_locs

        # these are the transformer layers that will run above the base model results, over each representation
        # of (text, image_i) for each image
        self._classifier_transformer_layers = classifier_transformer_layers
        self._classifier_transformer_layers_cls = classifier_transformer_layers_cls
        if classifier_transformer_layers:
            config = BertConfig.from_dict(config.to_dict())
            config.hidden_size = config.pooler_size
            config.num_attention_heads = 8
            w = torch.empty(config.pooler_size)
            nn.init.normal_(w, std=0.02)
            self.img_cls_token = nn.Parameter(w)
            self.agg_fc = nn.ModuleList(
                [BertLayer(config) for _ in range(classifier_transformer_layers)]
            )
        self.hidden_layer = nn.Sequential(
            nn.Linear(hid_dim * self.num_images, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
        )
        self.hidden_layer.apply(self.init_bert_weights)

        classifier_layer_input_size = hid_dim if classifier_transformer_layers_cls else hid_dim * 2
        self.classifier_layer = nn.Linear(classifier_layer_input_size, self.vocab.get_vocab_size("labels"))

        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

        self.accuracy = Average()
        self.tested_gen_accuracy = Average()
        self.accuracy_per_pattern = defaultdict(Average)
        self.accuracy_per_split = defaultdict(Average)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, visual_features, visual_positions, sentence, label=None, metadata=None):
        sentence = sum(zip(*((sentence,) * self.num_images)), ())
        batch_size, img_num, obj_num, feat_size = visual_features.size()
        assert img_num == self.num_images and obj_num == 36 and feat_size == 2048
        visual_features = visual_features.view(batch_size * self.num_images, obj_num, feat_size)
        visual_positions = visual_positions.view(batch_size * self.num_images, obj_num, self._num_vis_position_features)

        text_features = convert_sents_to_features(sentence, self.max_seq_length, self.tokenizer)
        input_ids = torch.tensor([f.input_ids for f in text_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in text_features], dtype=torch.long).cuda()

        x = self.encoder(input_ids, visual_features, visual_positions, attention_mask=input_mask)
        x = self.encode_step_2(x, batch_size)
        logit = self.classifier_layer(x)

        outputs = {}
        if label is not None:
            outputs["loss"] = self.loss(logit, label)
            predictions = logit.cpu().argmax(dim=1)
            labels = label.cpu()
            for i, md in enumerate(metadata):
                pattern = md['pattern']
                correct = predictions[i] == labels[i]
                self.accuracy_per_pattern[pattern](correct)
                if md['comp_splits']:
                    for comp_split in md['comp_splits']:
                        self.accuracy_per_split[comp_split](correct)
                if md['tested_comp_split']:
                    self.tested_gen_accuracy(correct)
                else:
                    self.accuracy(correct)

        if not self.training:
            outputs["predictions"] = logit.softmax(dim=-1)

        return outputs

    def encode_step_2(self, x, batch_size):
        if self._classifier_transformer_layers:
            x = x.view(batch_size, self.num_images, x.size(-1))
            input_images = self.num_images
            if self._classifier_transformer_layers_cls:
                x = torch.cat((
                    self.img_cls_token.expand(batch_size, 1, self.img_cls_token.shape[0]),
                    x
                ), dim=1)
                input_images += 1
            for layer_module in self.agg_fc:
                x = layer_module(x, torch.ones((batch_size, 1, 1, input_images)).cuda())
        if self._classifier_transformer_layers_cls:
            x = x[:, 0, :]  # take the "CLS" token
        else:
            x = x.view(-1, self.hid_dim * self.num_images)
            x = self.hidden_layer(x)
        return x

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            'accuracy': self.accuracy.get_metric(reset=reset),
            'gen_accuracy': self.tested_gen_accuracy.get_metric(reset=reset),
        }
        for pattern_index, metric in self.accuracy_per_pattern.items():
            metrics[f'_acc_pattern_{pattern_index}'] = metric.get_metric(reset=reset)
        for split, metric in self.accuracy_per_split.items():
            metrics[f'_acc_split_{split}'] = metric.get_metric(reset=reset)
        return metrics
