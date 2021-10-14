# Running COVR experiments

The following are instructions to replicate the experiments and results from the COVR paper.

1. Download and unzip the [COVR dataset](https://drive.google.com/file/d/10xlQ6isRdGX94BypoqN6klniGeqdLBJA/view?usp=sharing) into the `data` directory.
2. Download and unzip the image features of imSitu and Visual Genome scenes (to be available soon)  into the `data` directory.
3. Download [VisualBERT pre-trained model file](https://sid.erda.dk/share_redirect/GCBlzUuoJl) into the `pretrained_volta/volta_models` directory
   * (_this is only necessary if you want to train models starting from a pre-trained model_)
   * Links to additional Volta models are available in [Volta's repository](https://github.com/e-bug/volta/blob/main/MODELS.md).
     Models we tested with are: `VisualBERT (CTRL) - ctrl_visualbert_base` and `ViLBERT (CTRL) - ctrl_vilbert_base` (others should generally work as well)
   * you can also load the models from a different directory by setting the `--volta-path` argument of `train.py`
4. Install all required packages (use pytorch 1.8.x)
```
pip install -r requirements.txt
```

### Training an i.i.d experiment (starting from a pre-trained model)
Run this to train an experiment where the training and validation sets are from the same distribution and no compositional properties are used to filter
out examples.

```
python train.py
python train.py --text-baseline
```

### Training a compositional-split experiment (starting from a pre-trained model)
Here, we train the model without a subset of the training data that includes a specific property. We evaluate the model both on data from the development set that doesn't contain this property (for model selection) and the subset that does contain this property (to evaluate generalization).

```
python train.py --comp-split {PROPERTY}
python train.py --text-baseline --comp-split {PROPERTY}
```

Replace `{PROPERTY}` with the desired property you'd like to test on from the list below.

After each epoch, accuracies for all properties will be printed.

#### Available properties

```
answer_attribute
answer_number
has_attribute
has_compare
has_compare_more
has_complex_quantifier_scope
has_count
has_group_by
has_logic
has_logic_and
has_num_3
has_num_3_ans_3
has_quantifier
has_quantifier_all
has_same_attribute_color
lexical_1
lexical_2
lexical_3
number
number_2_answer_2
program_1
program_2
program_3
rm_v_c
tpl_choose_object
tpl_verify_attribute
tpl_verify_count_union_tpl_verify_count_group_by
tpl_verify_quantifier_attribute
```

combinations of two properties can also be used, spearated with `+`, for example:

```
has_complex_quantifier_scope+has_quantifier_all
has_count+has_attribute
has_count+rm_v_c
```

### Evaluating our fine-tuned models

You can evaluate our already fine-tuned models by running this command (remove `--cuda_device 0` if not using a GPU):

```
allennlp evaluate runs/{EXPERIMENT}/model.tar.gz ../data/val.jsonl --output-file metrics.json --include-package src --cuda-device 0 --batch-size 100
```

You should get the following results:
```
"accuracy": 0.6923523436366275
```

* [Download fine-tuned i.i.d VisualBERT model](https://drive.google.com/file/d/1CTHZLa30e_LrT8yYho0atp5iEErJbM5Q/view?usp=sharing)