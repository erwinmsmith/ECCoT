# ECCoT: Enhancing Effective Cognition via Chain of Thought in Large Language Models

## Project Overview
This repository contains the implementation of ECCoT (End-to-End Cognitive Chain of Thought Validation Framework), a framework designed to enhance the effective cognition capabilities of large language models (LLMs) by generating reliable thought chains. ECCoT addresses the challenges of reliability and interpretability in LLMs through structured reasoning chains and validation mechanisms.

## Key Features
- **End-to-End Reasoning Chain Validation**: Evaluates and refines reasoning chains in LLMs to improve interpretability and trustworthiness.
- **Markov Random Field-Embedded Topic Model (MRF-ETM)**: Identifies latent topic distributions in large-scale data for topic-aware chain-of-thought generation.
- **Causal Sentence-BERT (CSBert)**: Enhances causal relationship embedding to ensure the causality of reasoning chains.
- **Structured Ordering Statistics**: Filters ineffective reasoning chains to improve model performance.

## Technical Architecture
The ECCoT framework consists of several key components:
1. **Theme Recognition**: Uses MRF-ETM to identify topics and keywords in input texts.
2. **Theme Explanation**: Generates explanations based on recognized topics.
3. **Thought Cognition**: Creates reasoning chains based on topic-conditioned information.
4. **Effective Cognition**: Validates and filters effective cognitive processes using a rank framework.

## Getting Started

### Prerequisites
- Python 3.7+
- PyTorch
- Transformers
- Sentence-BERT
- NumPy
- Matplotlib

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/ECCoT.git
cd ECCoT
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Preparation
The framework is evaluated on three benchmark datasets:
- ANLI (Adversarial Natural Language Inference)
- SVAMP (Simple Variations on Arithmetic Math Word Problems)
- CommonQA (Commonsense Question Answering)

Place the datasets in the `data/` directory.

## Usage

### Running the Framework
Execute the following scripts in order:

1. Data processing:
```bash
python dataprocess.py
```

2. Main training:
```bash
python main_en.py
```

3. Reasoning quality assessment:
```bash
python reqa.py
```

4. Build chain of thought:
```bash
python buildcot.py
```

5. Build reasoning chains:
```bash
python buildcotreason.py
```

6. Build sentences:
```bash
python buildsentence.py
```

7. Fine-tune BERT:
```bash
python fbert.py
```

8. BERT fine-tuning:
```bash
python bert_ft.py
```

9. BERT embedding:
```bash
python bert_emb.py
```

10. Read numpy files:
```bash
python read_np.py
```

11. Build JSON files:
```bash
python buildjson.py
```

12. LLaMA factory fine-tuning:
```bash
python llamafactory.py
```

## Experimental Results

### Performance Comparison
The framework's performance across different datasets:

| Model          | ANLI  | SVAMP | CommonQA |
|----------------|-------|-------|----------|
| Step by Step   | 69.72 | 78.23 | 71.68    |
| Curation       | 67.39 | 76.86 | 79.26    |
| Expansion      | 66.73 | 78.21 | 78.66    |
| Feedback       | 66.07 | 75.53 | 77.66    |
| Self-Knowledge | 63.34 | 63.75 | 67.75    |
| Vanilla Fine-tuning | 64.59 | 63.85 | 64.62    |
| **ECCoT**      | **72.23** | **92.72** | **86.54** |

### Cognitive Process Effectiveness
Comparison of cognitive process effectiveness across datasets:

| Dataset | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---------|--------|---------|---------|---------|
| ANLI    | 63.17  | 62.31   | 40.01   | 44.65   |
| SVAMP   | 58.93  | 69.09   | 46.65   | 50.16   |
| CommonQA| 61.03  | 68.59   | 47.85   | 48.46   |

## References
For more details about the methodology and experimental setup, please refer to the original paper:
```
ECCoT: A Framework for Enhancing Effective Cognition via Chain of Thought in Large Language Models
Anonymous CogSci submission
```




## Contact
For questions or feedback, please contact 1262214827@qq.com.

---
