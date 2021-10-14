local number_images_per_question = 0;

{
    "dataset_reader": {
        "number_images_per_question": number_images_per_question
    },
    "model": {
        "num_images": number_images_per_question
    },
    "data_loader": {
        "batch_size": 120
    },
    "validation_data_loader": {
        "batch_size": 250
    },
    "trainer": {
        "num_epochs": 20
    }
}
