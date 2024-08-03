# DeBERTa-ELL: Automated Proficiency Assessment for English Language Learners

## Overview

DeBERTa-ELL is an advanced natural language processing (NLP) project that leverages the power of the DeBERTa (Decoding-enhanced BERT with Disentangled Attention) model to automatically assess the language proficiency of high school English Language Learners (ELLs) based on their essays. This project aims to provide a reliable, efficient, and scalable solution for educators and researchers in the field of second language acquisition and assessment.

## Features

- Utilizes state-of-the-art DeBERTa model for text analysis
- Assesses multiple aspects of language proficiency:
  - Cohesion
  - Syntax
  - Vocabulary
  - Phraseology
  - Grammar
  - Conventions
- Implements multi-label stratified k-fold cross-validation for robust model evaluation
- Supports both training and inference modes
- Includes data preprocessing and augmentation techniques
- Provides detailed logging and model checkpointing

## Requirements

- Python 3.10+
- PyTorch 2.3+
- Transformers 4.37+

For a complete list of dependencies, please refer to the `requirements.txt` file.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/arnavs04/deberta-ell.git
   cd deberta-ell
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

The training data is already in the `data/feedback-prize-english-language-learning/` directory.

### Training

To train the model, run:

```bash
python train.py
```

You can modify the hyperparameters in the `configs.py` file.

### Inference

To run inference on new data:

```bash
python inference.py
```

You can modify the hyperparameters in the `configs.py` file

## Model Architecture

This project uses the DeBERTa-v3-base model as the backbone for essay analysis. The model is fine-tuned on the task of multi-aspect proficiency assessment, with a custom head for multi-label regression.

## Performance

The performance of the model was evaluated using Smooth L1 Loss for training and validation, and Mean Column-wise Root Mean Square Error (MCRMSE) score for the final evaluation. Below are the summarized results for each fold:

| Fold     | Score   |  
| -------- | ------- |
| 0        | 0.4493  |
| 1        | 0.4576  |
| 2        | 0.4663  |
| 3        | 0.4529  |
| **Overall**  | **0.4566**  |

## Contributing

Contributions are welcomed to improve DeBERTa-ELL! Please feel free to submit issues, fork the repository and send pull requests!

## Citation

If you use this code for your research, please cite our project:

```
@software{DeBERTa_ELL2024,
  author = {Arnav Samal},
  title = {DeBERTa-ELL: Automated Proficiency Assessment for English Language Learners},
  year = {2024},
  url = {https://github.com/arnavs04/deberta-ell.git}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The authors of the DeBERTa paper: Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen
- The Hugging Face team for their excellent `transformers` library
- The organizers of the Feedback Prize - English Language Learning competition on Kaggle

## Contact

For any queries, please open an issue or contact [samalarnav@gmail.com](mailto:samalarnav@gmail.com).

