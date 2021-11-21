# Filtering overlapping distractors

This component is trained to predict overlapping terms such as "man" and "person",
"catching" vs "reaching for", etc., by fine-tuning a pre-trained transformers model on manually annotated data.

We train 3 separate models for each of the following types of overlapping terms: attributes ("dark" vs "black"),
nouns ("grass" vs "ground") and relations ("on" vs "on top of").


### Training

Use the following commands to train:

```
pip install -r requirements.txt

# attributes
allennlp train train_attributes.jsonnet -s experiments/attributes --force --include-package src

# nouns
allennlp train train_nouns.jsonnet -s experiments/nouns --force --include-package src

# relations
allennlp train train_relations.jsonnet -s experiments/relations --force --include-package src
```

### Predicting

And then the following code for prediction:

```
# attributes
predict experiments/attributes resources/attributes.test.csv --output-file experiments/attributes/attributes-predicted.csv --use-dataset-reader --dataset-reader-choice validation --include-package src --predictor term_predictor --batch-size 2000 --silent --cuda-device=0

# nouns
predict experiments/nouns resources/nouns.test.csv --output-file experiments/nouns/nouns-predicted.csv --use-dataset-reader --dataset-reader-choice validation --include-package src --predictor term_predictor --batch-size 2000 --silent --cuda-device=0

# relations
predict experiments/relations resources/relations.test.csv --output-file experiments/relations/relations-predicted.csv --use-dataset-reader --dataset-reader-choice validation --include-package src --predictor term_predictor --batch-size 2000 --silent --cuda-device=0
```