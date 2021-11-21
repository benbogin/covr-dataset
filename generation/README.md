# Generating COVR examples

Generating the entire COVR dataset was a rather complex process with multiple moving parts, thus it might take some time until we
upload all the code that we have used. If there's anything in particular that's missing and would be
helpful for something you're working on, please don't hesitate to reach out!

For now, the following is available:
1. [**Filtering overlapping distractors**](auxiliary_models/overlapping_distractors) - this component is trained to predict overlapping terms such as "man" and "person",
   "catching" vs "reaching for", etc., by fine-tuning a pre-trained transformers model on manually annotated data.