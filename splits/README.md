# Compositionally Splitting COVR

Using this code, you can create new compositional splits of COVR.

To run our default splits used in COVR paper, use the following command:
```
python split.py
```
The script performs the following actions:
1. Loads the dataset from default path `../experiments/data/` (configurable using `--input-ds-path`)
2. Creates the split files `../experiments/data/updated_dataset` (configurable using `--output-ds-path`).
   The files that are created are:
   * `{SPLIT_NAME}_train_iid.json`: All ids of examples from the training set which _do not_ contain the compositional split, generally used for training
   * `{SPLIT_NAME}_train_comp.json`: All ids of examples from the training set which _do_ contain the compositional split, generally discarded
   * `{SPLIT_NAME}_val_iid.json`:  All ids of examples from the validation set which _do not_ contain the compositional split, generally discarded
   * `{SPLIT_NAME}_val_comp.json`: All ids of examples from the validation set which _do_ contain the compositional split, generally used for compositional evaluation 
   * `{SPLIT_NAME}_val_iid_examples.json`: A subset of examples for the matching file, helpful for debugging
   * `{SPLIT_NAME}_val_comp_examples.json`: A subset of examples for the matching file, helpful for debugging 

The above script will create a file for each split; but if you want to create new dataset files with the same format as the existing files
(i.e. `train.jsonl`/`val.jsonl`), which will generally be needed for training, run the following command:
```
python prepare_dataset.py
```

### Creating new splits
If you'd like to experiment with new compositional splits, simply implement a new `Splitter` class ([defined here](splitters/splitter.py)).

You will need to implement `get_split(qst)` which returns the string `train` or `test` according to the properties of the given
example, and if necessary, `fit(*lines)` which accepts all training examples, grouped by `train`/`val` sets.

Finally, add your new splitters to `split.py` and `prepare_dataset.py`.