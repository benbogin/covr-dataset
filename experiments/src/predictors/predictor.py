from typing import List, Dict

from allennlp.common import JsonDict
from allennlp.common.util import sanitize
from allennlp.data import DatasetReader, Instance, Vocabulary, allennlp_collate
from allennlp.data.fields import LabelField, MetadataField
from allennlp.models import Model
from allennlp.predictors import Predictor
from allennlp.data import TensorDict


@Predictor.register("vcqa", exist_ok=True)
class VCQAPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader):
        super().__init__(model, dataset_reader)

        self._dataset_reader._always_load_features = True

    @staticmethod
    def prepare_batch_output(vocab: Vocabulary, outputs: List[Dict], batch_input: List[TensorDict]):
        for instance, output in zip(batch_input, outputs):
            instance = VCQAPredictor._remove_fields(instance)
            output['sentence'] = instance['sentence']
            output['scenes'] = instance['metadata']["scenes"]
            output['qid'] = instance['metadata']["qid"]
            output['pattern_index'] = instance['metadata']["pattern_index"]
            output['answer'] = instance['label']

            predictions_probs = output['predictions']
            top_predictions = predictions_probs.argsort()[::-1][:5]
            output['predictions'] = []
            for pred in top_predictions:
                output['predictions'].append((
                    vocab.get_token_from_index(pred, "labels"),
                    predictions_probs[pred],
                ))
            output['prediction'] = output['predictions'][0][0]
            if output.get('answer'):
                output['correct'] = output['prediction'] == output.get('answer')
        return sanitize(outputs)

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        for instance in instances:
            self._dataset_reader.apply_token_indexers(instance)
        outputs = self._model.forward_on_instances(instances)

        return self.prepare_batch_output(self._model.vocab, outputs, instances)

    @staticmethod
    def _remove_fields(instance):
        if type(instance) is Instance:
            output = {}
            for key, value in instance.fields.items():
                if type(value) is LabelField:
                    output[key] = value.label
                elif type(value) is MetadataField:
                    output[key] = value.metadata
            return output
        else:
            return instance
