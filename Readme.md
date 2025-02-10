# Time Series Prediction (Generation) Diffusion Model

A conditional diffusion model for time series (stock market) data visualization and prediction using PyTorch. This project implements a DDIM-based approach to analyze and generate stock market patterns, with a focus on volume and price movements.

## Overview

This project consists of two main components:
1. Data preprocessing and image generation from stock market time series
2. A conditional diffusion model for learning and generating stock market patterns

### Features

- Conditional diffusion model with UNet architecture
- DDIM sampling for efficient inference
- Multi-channel stock data visualization
- Attention mechanism for better pattern recognition
- Flexible GPU/CPU processing support
- Custom dataset handling for stock market data

## Requirements

```
python >= 3.8
torch >= 1.9.0
torchvision
numpy
pandas
matplotlib
pillow
GPUtil
tqdm
```

## Project Structure

```
├── stock_images/          # Generated stock market images
├── checkpoints/          # Model checkpoints
├── data/                 # Raw stock market data
│   └── master_data_IBM.csv
├── test_results/        # Model evaluation results
│   ├── comparisons/     # Original vs predicted comparisons
│   └── intermediate_steps/  # Visualization of sampling steps
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd DDIM_Time_Series
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

The model expects stock market data in CSV format with the following columns:
- timestamp
- close (price)
- volume

To prepare your data:

1. Place your CSV file in the `data/` directory
2. Run the data preprocessing script:
```python
python dataset_creation.py
```

This will generate RGB images where:
- Red channel: Volume data
- Green channel: Price changes before cutoff time
- Blue channel: Price changes after cutoff time

## Model Architecture

The model uses a UNet architecture with:
- Conditional diffusion process
- Time embeddings
- Attention blocks
- Skip connections
- Group normalization

### Key Components:

1. **UNet**: Main architecture for the diffusion process
2. **AttentionBlock**: Self-attention mechanism for capturing long-range dependencies
3. **Block**: Basic building block with residual connections
4. **SinusoidalPositionEmbeddings**: Time step embeddings

## Training

To train the model:

```python
python trainDDIM.py 
```

Key parameters:
- `batch_size`: Number of images per batch
- `image_size`: Size of input images
- `n_epochs`: Number of training epochs
- `device`: "GPU" or "CPU"

The model automatically saves checkpoints every 10 epochs in the `checkpoints/` directory.



## Model Parameters

- `n_timesteps`: 1000 (diffusion steps)
- `base_channels`: 64
- `channel_mults`: (1, 2, 4, 8)
- `attention_resolutions`: (8, 16)
- `time_emb_dim`: 256

## Performance Optimization

The code includes automatic GPU selection based on:
- Available memory
- Current GPU load
- Memory utilization

To optimize performance:
1. Adjust batch size based on available GPU memory
2. Modify number of sampling steps for inference speed/quality trade-off
3. Use `eta` parameter in DDIM sampling to control stochasticity

## Memory Management

The model implements several memory optimization techniques:
- Gradient checkpointing (optional)
- Efficient attention computation
- Automatic batch size adjustment
- Checkpoint management with rolling window

## Troubleshooting

Common issues and solutions:

1. **GPU Out of Memory**
   - Reduce batch size
   - Decrease image size
   - Use gradient checkpointing

2. **Data Loading Errors**
   - Verify CSV file format
   - Check for missing values
   - Ensure correct time format

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License
CC BY-NC: Non-commercial use with attribution

## Acknowledgments

This project builds upon several key papers and implementations:
- DDIM (Denoising Diffusion Implicit Models)
- U-Net architecture
- Attention mechanisms in deep learning

