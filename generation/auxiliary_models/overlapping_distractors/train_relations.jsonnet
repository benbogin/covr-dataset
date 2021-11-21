local model_name = "roberta-large-mnli";

{
    "train_data_path": 'resources/relations.train.csv',
    "validation_data_path": 'resources/relations.val.csv',
    "dataset_reader": {
        "type": "terms",
        "source_tokenizer": {
            "type": "pretrained_transformer",
            "model_name": model_name
        },
        "source_token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": model_name,
                "namespace": "tokens"
            }
        }
    },
    "model": {
        "type": "term_pretrained_for_classification"
    },
    "data_loader": {
        "batch_size": 150,
        "shuffle": true
    },
    "trainer": {
        "num_epochs": 50,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 3e-5
        },
        "validation_metric": "+accuracy",
        "checkpointer": {
            "num_serialized_models_to_keep": 0
        }
    }
}
