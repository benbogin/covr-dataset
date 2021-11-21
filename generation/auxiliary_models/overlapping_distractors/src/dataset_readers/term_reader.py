import logging
from random import Random
from typing import Dict, List

import csv
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer, Token
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("terms")
class Seq2SeqDatasetReader(DatasetReader):
    def __init__(
        self,
        source_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None
    ) -> None:
        super().__init__()
        self._source_tokenizer = source_tokenizer or SpacyTokenizer()
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}

        self._random = Random(0)

        self._data = []
        self._shuffle_pair = False

    @overrides
    def _read(self, file_path: str):
        is_val = '.val.csv' in file_path
        is_test = '.test.csv' in file_path
        file_path = file_path.replace('.train.csv', '.csv')
        file_path = file_path.replace('.val.csv', '.csv')
        file_path = file_path.replace('.test.csv', '.csv')

        count = 0

        if not self._data:
            self._data = list(enumerate(csv.reader(open(file_path, 'rt'))))

        if "attributes" in file_path or "nouns" in file_path:
            self._shuffle_pair = not is_val and not is_test
        else:
            self._shuffle_pair = False

        if not is_test:
            self._random.shuffle(self._data)

        for i, line in self._data:
            if not is_test:
                if i == 0:
                    continue
                if i % 5 == 0:
                    # use a randomly picked 20% of the data for validation
                    if not is_val:
                        continue
                else:
                    if is_val:
                        continue
            if "attributes" in file_path:
                first, second, examples, cnt, label1 = line
                label2 = None
            elif "relations" in file_path:
                first, second, examples, cnt, _, label1, label2 = line
            elif "nouns" in file_path:
                first, second, examples, cnt, _, label1, _ = line
                label2 = None

            if (not is_test and label1) or is_test:
                count += 1
                yield from self.text_to_instance(first, second, label1, label2)

    @overrides
    def text_to_instance(
        self,
        first: str,
        second: str,
        label1: str,
        label2: str,
    ) -> List[Instance]:
        """
        first and second are the two terms we want to train on.
        label1 and label2 are the labels - in some cases the labelling is symmetric (thus label2 will remain None) and
        in other cases it will not be symmetric, that is, the labels depend on the order of `first` and `second`.
        """
        instances = []
        first = first.replace(":", " ").replace("_", "")
        second = second.replace(":", " ").replace("_", "")
        attrs = [first, second]
        if self._shuffle_pair:
            self._random.shuffle(attrs)
        tokenized_source = self._source_tokenizer.tokenizer.tokenize(attrs[0], attrs[1], add_special_tokens=True)
        source_field = TextField([Token(t) for t in tokenized_source], self._source_token_indexers)

        # roberta-nli is trained with 0 for contradiction / 1 for neutral
        if label1 in ["0", "1"]:
            label1_val = 0 if label1 == '1' else 1
        else:
            label1_val = 0 if label1 == 'c' else 1
        label_field = LabelField(label1_val, skip_indexing=True)

        metadata = {"first": first, "second": second, "label": label1_val}
        if label1 != "":
            metadata["manual_label"] = label1_val

        fields = {
            "tokens": source_field,
            "label": label_field,
            "metadata": MetadataField(metadata),
        }

        instances.append(Instance(fields))

        if label2 is not None:
            tokenized_source = self._source_tokenizer.tokenizer.tokenize(attrs[1], attrs[0], add_special_tokens=True)
            source_field = TextField([Token(t) for t in tokenized_source], self._source_token_indexers)

            # roberta-nli is trained with 0 for contradiction / 1 for neutral
            if label2 in ["0", "1"]:
                label2_val = 0 if label2 == '1' else 1
            else:
                label2_val = 0 if label2 == 'c' else 1
            label_field = LabelField(label2_val, skip_indexing=True)

            metadata = {"first": second, "second": first, "label": label2_val}
            if label2 != "":
                metadata["manual_label"] = label2_val

            fields = {
                "tokens": source_field,
                "label": label_field,
                "metadata": MetadataField(metadata),
            }

            instances.append(Instance(fields))

        return instances

