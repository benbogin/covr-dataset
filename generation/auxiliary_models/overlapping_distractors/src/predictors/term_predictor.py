import csv
import io
from typing import List

import numpy as np

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register("term_predictor")
class TermPredictor(Predictor):
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        for instance in instances:
            self._dataset_reader.apply_token_indexers(instance)
        outputs = self._model.forward_on_instances(instances)

        for instance, output in zip(instances, outputs):
            output['prediction'] = np.argmax(output['probs'])

            # if this row was manually annotated, take the manual annotation instead of the prediction
            if 'manual_label' in instance['metadata'] and instance['metadata']['manual_label'] in ["0", "1", 0, 1]:
                output['prediction'] = int(instance['metadata']['manual_label'])
        return sanitize(outputs)

    def dump_line(self, outputs: JsonDict) -> str:
        values = [outputs['metadata']['first'], outputs['metadata']['second'], outputs['prediction']]
        line = io.StringIO()
        writer = csv.writer(line)
        writer.writerow(values)
        csvcontent = line.getvalue()
        return csvcontent
