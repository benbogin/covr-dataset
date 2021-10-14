local number_images_per_question = 5;

{
    "dataset_reader": {
        "type": "vcqa",
        "features_h5_path": "data/gqa_imsitu_features.h5",
        "img_key_to_index_path": "data/gqa_imsitu_key_to_index.json",
        "number_images_per_question": number_images_per_question,
        "downsample": null
    },
    "train_data_path": "train.jsonl",
    "validation_data_path": "val.jsonl",
    "test_data_path": "test.jsonl",
    "evaluate_on_test": true,
    "model": {
        "type": "volta",
        "max_seq_length": 60,
        "num_images": number_images_per_question,
        "pretrained_model_path": "data/model",
        "classifier_transformer_layers": 2,
        "classifier_transformer_layers_cls": true
    },
    "data_loader": {
        "type": "vcqa_data_loader",
        "batch_size": 12,
        "num_workers": 1
    },
    "validation_data_loader": {
        "type": "vcqa_data_loader",
        "batch_size": 120,
        "num_workers": 1
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "eps": 1e-6,
            "weight_decay": 0.001,
            "parameter_groups": [
                [["bias", "LayerNorm.bias", "LayerNorm.weight"], {"weight_decay": 0.0}],
            ]
        },
        "grad_clipping": 1.0,
        "learning_rate_scheduler": {
            "type": "linear_with_warmup_float"
        },
        "num_epochs": 8,
        "cuda_device": 0,
        "validation_metric": "+accuracy",
        "checkpointer": {
            "num_serialized_models_to_keep": 1
        }
    }
}
