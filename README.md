# CodeComplex: Dataset for Worst-Case Time Complexity Prediction

CodeComplex is a comprehensive dataset designed to evaluate Large Language Models (LLMs) in the complex task of worst-case time complexity prediction for code. The dataset features **4,900 Java codes** and **4,900 Python codes**, annotated with detailed complexity labels across seven distinct classes. It provides a robust benchmark for advancing the reasoning capabilities of LLMs in software development.

---

## 📜 Features

- **Bilingual Dataset**: Includes both Java and Python codes from competitive programming platforms.
- **Seven Complexity Classes**: Annotated across constant (O(1)), linear (O(n)), quadratic (O(n)), cubic (O(n³)), logarithmic (O(log n)), linear-logarithmic (O(n log n)), and exponential.
- **Balanced Class Distribution**: Designed to mitigate biases and enhance model generalization.
- **Comprehensive Annotations**: Labeled by expert annotators considering input characteristics, library impacts, and control structures.
- **Novel Evaluation Metric**: Introduces the Hierarchy Complexity Score (HC-Score) for nuanced performance assessment.

---

## 📊 Dataset Statistics

| Complexity Class | Java Codes | Python Codes |
|-------------------|------------|--------------|
| O(1)             | 750        | 791          |
| O(n)             | 779        | 853          |
| O(n)            | 765        | 657          |
| O(n³)            | 601        | 606          |
| O(log n)         | 700        | 669          |
| O(n log n)       | 700        | 796          |
| Exponential      | 605        | 528          |

---

## 🔍 Why CodeComplex?

1. **Fill the Gap**: Addresses limitations of existing benchmarks like CoRCoD and TASTY.
2. **Detailed Complexity Analysis**: Accounts for input representation and algorithmic structures.
3. **Open Source**: Fully accessible for research and development.
4. **Diverse Applications**: Useful for NLP, Software Engineering, and Programming Language communities.

---

## 🛠 Getting Started

### Installation

Clone the repository:
```bash
git clone https://github.com/sybaik1/CodeComplex-Data.git

@article{CodeComplex2024,
  author    = {Seung-Yeop Baik and others},
  title     = {CodeComplex: Dataset for Worst-Case Time Complexity Prediction},
  year      = {2024},
  url       = {https://github.com/sybaik1/CodeComplex-Data},
}

