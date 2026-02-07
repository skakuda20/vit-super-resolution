# Image Super Resolution with Deep Learning

This project implements image super-resolution using deep learning techniques. It includes models, datasets, and utilities to train, evaluate, and test super-resolution models.

## Overview of EnhanceNetViT

The `EnhanceNetViT` model combines the strengths of convolutional neural networks (CNNs) and vision transformers (ViTs) to achieve high-quality image super-resolution. It leverages:

- **Convolutional Layers**: feature extraction and spatial detail preservation
- **Residual Blocks**: feature learning and mitigate vanishing gradient issues
- **Vision Transformers**: capture long-range dependencies and global context in images
- **Pixel Shuffle Upsampling**: efficient and artifact-free upscaling of low-resolution images

This hybrid architecture enables the model to produce sharp and detailed super-resolution outputs while maintaining computational efficiency.

## How to Use

### Setting Up the Environment

Before running the project, set up a Python virtual environment and install the required dependencies:

1. **Create a virtual environment**:

   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment**:

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   Ensure that the `requirements.txt` file is located in the root directory of the project.

### Research
This project involved researching SRCNN and base EnhanceNet models which are contianed within Jupyter Notebooks in the `research/` directory.

### Download DIV2K Dataset
This project is based on the DIV2K Dataset and can be downloaded by running the `download_data.py`. Set the respective paths to the train and validate directories in the `params.yaml` file.

### Running the Project

The main entry point for running the project is the `run.py` module located in the `src/` directory. This script orchestrates the training and evaluation of the super-resolution models.

To execute the script, use the following command:

```bash
python src/run.py
```

### Configuring Parameters

The `params.yaml` file in the `src/` directory contains all the configurable parameters for the project. These parameters include settings for training, evaluation, and model architecture.

#### Example Parameters in `params.yaml`:

```yaml
learning_rate: 0.001
batch_size: 16
num_epochs: 50
model: EnhanceNetViT
dataset_path: data/DIV2K
```

#### Modifying Parameters

To customize the behavior of the project, edit the `params.yaml` file. For example:

- Change `learning_rate` to adjust the optimizer's learning rate.
- Modify `batch_size` to control the number of samples processed per training step.
- Update `dataset_path` to point to a different dataset location.

### Running with Custom Parameters

The `run.py` script automatically reads the parameters from `params.yaml`. Ensure the file is correctly configured before running the script.

```bash
python src/run.py
```

### Outputs

The script generates the following outputs:

- Trained model weights saved in the `models/` directory.
- Evaluation metrics printed to the console and optionally saved to a log file.
- Super-resolved images saved in the `results/` directory (if applicable).

For more details, refer to the comments in the `run.py` script and the structure of the `params.yaml` file.
