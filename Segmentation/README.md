# Environment Parameters

Below is a table listing all needed environment parameters to run dataset preparation, training and evaluation scripts and notebooks.

| Parameter                        | Description                                                |
| -------------------------------- | ---------------------------------------------------------- |
| SEED                             | Random seed for reproducibility                            |
| RAW_DATA_DIR                     | root path of the raw original images                       |
| DATASETS_DIR                     | root path of processed datasets |
| MODEL_CHECKPOINTS_DIR            | directory where (foundation) model checkpoints are stored for import (e.g. SAM checkpoints)                     |
| MODEL_OUT_DIR                    | root path for storing trained model parameters        |
| WANDB_DIR                        | path to Weights & Biases metadata directory |
| LOG_DIR                          | directory for training and debug logs                      |
| DATASET_NAME                     | dame of the dataset used for training        |
| HIGHRES_IMG_DIR_NAME             | Subfolder for 4096×4096 images                             |
| HIGHRES_MASK_DIR_NAME            | Subfolder for 4096×4096 masks                              |
| LOWRES_IMG_DIR_NAME              | Subfolder for 1024×1024 images                             |
| LOWRES_MASK_DIR_NAME             | Subfolder for 1024×1024 masks                              |
| CROPPED_AUG_IMG_DIR_NAME         | Subfolder for 256×256 cropped (and augmented) images                     |
| CROPPED_AUG_MASK_DIR_NAME        | Subfolder for 256×256 cropped (and augmented) masks                      |
| SAM3_OUTPUT_DIR_NAME             | Subfolder containing SAM3 output (zero-shot segmentations used for annotation)                           |
| CVAT_DIR_NAME                    | Directory for exported CVAT projects                       |
| CVAT_BACKGROUND_COLOR            | CVAT background label RGB value                     |
| CVAT_MASK_COLOR                  | CVAT mask label RGB value                    |
| CVAT_GENERATE_BW_MASKS           | Whether to convert masks to binary black/white (0/255), instead of 0/1             |
| DATASET_LOWRES_RESIZE            | shape to which images are resized during dataset generation (before cropping)                    |
| DATASET_GAUSSIAN_NOISE           | Gaussian noise standard deviation                          |
| DATASET_GAMMA_RANGE              | Gamma augmentation range                                   |
| DATASET_MAX_DISTORT              | Maximum elastic distortion                                 |
| DATASET_DISTORT_GRID_SIZE        | Grid size for elastic distortion                           |
| DATASET_RANDOM_CROP_SIZE         | Size of random crops taken on original 4096x4096 images                              |
| USE_WANDB                        | Enables or disables Weights & Biases tracking              |
| WANDB_ENTITY                     | W&B team, should be EM_IMCR_BIOVSION                                      |
| WANDB_PROJECT                    | W&B project name                                           |
| WANDB_API_KEY                    | API key for W&B (keep private)                             |
| SAM_LORA_VENV                    | Virtual environment path for SAM LoRA finetuning                   |
| SAM_LORA_FINETUNE_IMAGE_ENCODER  | Whether to fine-tune the image encoder                     |
| SAM_LORA_FINETUNE_MASK_DECODER   | Whether to fine-tune the mask decoder                      |
| SAM_LORA_FINETUNE_PROMPT_ENCODER | Whether to fine-tune the prompt encoder                    |
| SAM_LORA_USE_CROPPED_IMAGES      | Whether to use cropped 256×256 images (cropped from resized images in a grid, so 1024x1024 images would result in 16 crops)                      |
| SAM_LORA_LR                      | Learning rate                                              |
| SAM_LORA_NUM_CLASSES             | Number of segmentation classes, 1 for semantic segmentation (background/foreground)                             |
| SAM_LORA_BATCH_SIZE              | Batch size                                                 |
| SAM_LORA_MAX_EPOCHS              | Maximum number of epochs                                   |
| SAM_LORA_UPSAMPLE_LOWRES_LOGITS  | Size to which low-resolution (256x256) logits output by SAM are resized during loss calculation. Leave empty for no upsampling, meaning the ground truth images will be resized to 256x256                |
| SAM_LORA_MODEL_CHECKPOINT        | SAM checkpoint used for training (located in MODEL_CHECKPOINTS_DIR)                      |
| SAM_LORA_MODEL_TYPE              | SAM model type (e.g., vit_b)                               |
| SAM_LORA_RANK                    | LoRA rank value                                            |
| EARLY_STOPPING_PATIENCE          | Number of epochs without improvement before stopping       |
| EARLY_STOPPING_DELTA             | Minimum improvement considered progress                    |
| EARLY_STOPPING_MIN_EPOCHS        | Minimum epochs before early stopping can activate          |
| HUGGINGFACE_TOKEN                | Access token for Hugging Face models (keep private)        |
