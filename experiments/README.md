# Running COVR experiments

The following are instructions to replicate the experiments and results from the COVR paper.

1. Download and unzip the COVR dataset into the data directory.
2. Download and unzip the image features of imSitu and Visual Genome scenes.
3. Install all required packages
```
pip install -r requirements.txt
```

### Running an i.i.d experiment
(where training and validation sets are from the same distribution and no compositional properties are used to filter
out examples)

```
cd src
python train.py
python train.py --text-baseline
```