# plamtral

PLaMTraL - A transfer learning library for pre-trained language models.

## Installation

Install plamtral with pip:

```bash
  pip install plamtral
```

## Requirements

* torch 1.12.1
* tqdm 4.64.1
* transformers 4.24.0
* nltk 3.7
* torchmetrics
    
## Features
### Fine-tuning
Fine-tuning large pretrained language models on downstream tasks remains
the de-facto learning paradigm in NLP. However, several fine tuning approaces exist other than the usual vanilla variant, which can be more effective or efficient. The fine tuning techniques provided in this package are:
- **BitFit** - a sparse fine tuning method where only the bias terms of the model (or a subset of them) are being modified. Reference: https://arxiv.org/pdf/2106.10199.pdf.
- **Chain thaw** - an approach that sequentially unfreezes and fine-tunes a single layer at a time. Reference: https://arxiv.org/pdf/1708.00524.pdf.
- **ULMFiT** - an effective transfer learning method that introduces techniques (slanted triangular learning rate, disciminative fine-tuning, and gradual unfreezing) that are key for fine-tuning a language model. Reference: https://arxiv.org/pdf/1801.06146.pdf.
- **Vanilla fine tuning** - the standard fine-tuning approach (fine-tune the whole model, fine-tune the last n layers, or fine-tune a specific layer).
### Parameter efficient approaches
Since conventional fine-tuning approaches can become expensive as they often require the storage of a large number of parameters, recent work has proposed a variety of parameter-efficient transfer learning methods that only fine-tune a small number of (extra) parameters to attain strong performance. The parameter efficient techniques provided in this package use:
- **AdapterDrop** - an approach that removes adapters (see Houlsby Adapter) from lower transformer layers during training and inference. Reference: https://arxiv.org/pdf/2010.11918.pdf.
- **Bapna Adapters** - a variant of the Houlsby Adapter (see Houlsby Adapter). Reference: https://arxiv.org/pdf/1909.08478.pdf.
- **Houlsby Adapters** - compact, trainable modules which are inserted between layers of a pre-trained network. Reference: https://arxiv.org/pdf/1902.00751.pdf. 
- **LoRA** - an approach that freezes the pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture. Reference: https://arxiv.org/pdf/2106.09685.pdf.
- **MAM Adapters** - an approach that attempts to combine the design elements of several parameter efficient approaches in order to arrive at a unified framework. Reference: https://arxiv.org/pdf/2110.04366.pdf.
- **Parallel Adapters** - Houlsby Adapters which are inserted in a parallel manner rather than serially. Reference: https://arxiv.org/pdf/2104.08154v1.pdf.
- **Prefix Tuning** - an approach that keeps language model parameters frozen, but optimizes a small continuous task-specific vector (called the prefix). Reference: https://arxiv.org/pdf/2101.00190.pdf.
- **Prompt Tuning** - an approach that freezes the entire pre-trained model and only allows an additional k tunable tokens per downstream task to be prepended to the input text. Reference: https://aclanthology.org/2021.emnlp-main.243.pdf.
## Usage/Examples
To use a GPT2 model with Houlsby Adapters (for example):
```python
from parameter_efficient.adapter import Model_with_adapter
from tl_lib.utils import load_dataloaders
from tl_lib.tl_train import train

# Load the GPT2 model with Houlsby Adapters
model_obj = Model_with_parallel_adapter('GPT2')
# Create the train, validation and test dataloaders from the dataset file
train_loader, val_loader, test_loader = load_dataloaders('GPT2', dataset_path='path/to/dataset_file')
# Train the model
train(model_obj, train_loader, val_loader, verbose = True, model_save_name = 'path/to/model')
```


## Authors

[@Vibhu04](https://www.github.com/Vibhu04)


## License

[MIT](https://choosealicense.com/licenses/mit/)

