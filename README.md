# CodeComplex: A Dataset for the Worst-Case Time Complexity Prediction


## TL; DR

CodeComplex is a comprehensive dataset designed to evaluate models
in the complex task of predicting the worst-case time complexity for codes.
The dataset features **4,900 Java codes** and **4,900 Python codes**,
annotated with complexity labels across seven distinct classes.
It provides a robust benchmark for recognizing the reasoning capabilities
of LLMs in software development.

## About CodeComplex

CodeComplex is a dataset about the worst-case time complexities of codes.
The dataset is designed to evaluate the reasoning abilities
of Large Language Models in predicting the time complexity of code.
With a bilingual focus on Java and Python,
this dataset addresses gaps in previous benchmarks,
offering balanced complexity classes,
extensive annotation, and novel evaluation metrics.

CodeComplex leverages competitive programming submissions from Codeforces
and builds on the CodeContests dataset developed by DeepMind.
It extends the CoRCoD dataset by enhancing class distribution and introducing bilingual support.

### Annotation Process

The annotation process was conducted by a team of three experts with over ten years of experience.
Independent annotations were followed by cross-validation to ensure consistency.

### Hierachy Complexity Score (HC-score)

HC-Score penalizes predictions based on their deviation from correct complexity classes.
It is calculated as follows:

$$HC-Score(P, R) = (Σ |p_i - r_i|) / (N × Number of Classes)$$

Here,  represents predicted complexity classes,  represents true complexity classes, and  is the total number of predictions. The Windowed HC-Score expands this metric by allowing flexible evaluation within a defined margin of error:

$$HC-Score_w(P, R) = Σ max(1 - |p_i - r_i| / w, 0) / N$$

This nuanced approach factor in the partial reasoning capabilities of LLMs.

## 📜 Features

- **Bilingual Dataset**: Includes both Java and Python codes from competitive programming platforms.
- **Seven Complexity Classes**: Annotated across constant (O(1)), linear (O(n)), quadratic (O(n)), cubic (O(n³)), logarithmic (O(log n)), linear-logarithmic (O(n log n)), and exponential.
- **Balanced Class Distribution**: Designed to mitigate biases and enhance model generalization.
- **Comprehensive Annotations**: Labeled by expert annotators considering input characteristics, library impacts, and control structures.
- **Novel Evaluation Metric**: Introduces the Hierarchy Complexity Score (HC-Score) for nuanced performance assessment.



## 📊 Dataset Statistics

| Complexity Class | Java Codes | Python Codes |
|-------------------|------------|--------------|
| O(1)             | 750        | 791          |
| O(n)             | 779        | 853          |
| O(n^2)            | 765        | 657          |
| O(n^3)            | 601        | 606          |
| O(log n)         | 700        | 669          |
| O(n log n)       | 700        | 796          |
| Exponential      | 605        | 528          |
| Total            | 4900       | 4900         |



## 🔍 Why CodeComplex?

1. **Fill the Gap**: Addresses limitations of existing benchmarks like CoRCoD and TASTY.
2. **Detailed Complexity Analysis**: Accounts for input representation and algorithmic structures.
3. **Open Source**: Fully accessible for research and development.
4. **Applications**: Useful for NLP, Software Engineering, and Programming Language communities.


## Annotator Guideline

1. Instructions

Check the Variables Described in the Algorithm Problems

Each algorithm implementation can have many variable instances. Only consider the variables that are given as inputs from the problems for calculating the time complexity.

* **Input Variable Notation**

For convenience, use n and m to denote input variables and |n| and |m| to denote the size of n and m.

2. Time Complexity Calculation

Based on the input variables, follow the instructions below to calculate the time complexity:

a. Single Numeric Input

When only a number n is given as an input, calculate the time complexity proportional to n.

Do the same for multiple variables. For example, when only n is given as an input, the variable used to denote the time complexity of the code is n.

b. Multiple Numeric Inputs

When n and m numeric instances are given as inputs, calculate the time complexity proportional to the one with higher complexity.

For example, if m = n^2, compute the complexity of the code with m. If the implemented algorithm runs in O(n^2) = O(m), it belongs to the linear complexity class.

c. Constant Inputs

If the input is given as constant values, the complexity of the code belongs to the constant class.

For instance, if an algorithm problem states that exactly 3 numeric values are given as inputs, and the solution code only uses a constant number of operations, the code belongs to the constant complexity class.

3. Input Constraints

Consider cases where the code utilizes the input constraints of the problem. For example, if the input is given as n ≤ a, the code can use the fixed value a instead of n. Mark these codes as unsuitable.

4. Built-in Library Usage

Consider built-in libraries used in the algorithm (e.g., HashMap, sort, etc.) when calculating the time complexity of the entire code.

For instance, given n numeric instances as inputs, if an algorithm uses O(n) iterations of a built-in sort algorithm, the time complexity for the algorithm is O(n^2 ∗ log(n)).

5. Unreachable Code

When the code has unreachable sections, only consider the reachable code for time complexity analysis.

6. Mark Unsuitable Items

Mark items that do not belong to any of the 7 predefined complexity classes.


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

