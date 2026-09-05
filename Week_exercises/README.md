# IN4080 – Weekly Exercises

This directory contains my work on the weekly exercises for **IN4080 – Natural Language Processing** at the University of Oslo (UiO), Autumn 2026.

The notebooks include my solutions, experiments, notes, and code written while working through the course exercises.

## Contents

### Week 1 – Text Preprocessing and Word Frequency Analysis

The first exercise set introduces basic NLP preprocessing and text analysis using *Peter Pan* as the example corpus.

Topics covered:

* Working with Jupyter Notebook
* Loading and inspecting text data
* Text preprocessing with **spaCy**
* Sentence segmentation
* Cleaning text using regular expressions
* Tokenization
* Counting word frequencies with `Counter`
* Tokens and types
* Type-token ratio
* Hapax legomena
* Working with **Pandas DataFrames**
* Visualizing word-frequency distributions with **Matplotlib**
* Exploring Zipf's law

Files:

```text
week1/
├── Ex1.py
├── in4080-exercises1.ipynb
├── peter-pan.txt
├── slides_week1.pdf
└── README.md
```

---

### Week 2 – Preprocessing for Text Classification

The second exercise set introduces the basic preprocessing pipeline used for text classification.

The exercises use a **Movie Reviews** dataset containing positive and negative movie-review texts.

Topics covered:

* Loading CSV data with **Pandas**
* Inspecting datasets
* Separating input features (`X`) and labels (`y`)
* Splitting data into:

  * 80% training set
  * 10% validation set
  * 10% test set
* Using `train_test_split` from **Scikit-learn**
* Text tokenization with `CountVectorizer`
* Creating **Bag-of-Words (BoW)** representations
* Learning a vocabulary from training data
* Transforming validation and test data using the learned vocabulary
* Understanding feature matrices for text classification

Files:

```text
week2/
├── in4080-exercises2.ipynb
├── movie_review.csv
└── README.md
```

---

## Main Libraries

The exercises currently use:

```text
Python
Jupyter Notebook
spaCy
NumPy
Pandas
Matplotlib
Scikit-learn
```

Some exercises also make use of Python's built-in libraries such as:

```python
re
collections
string
```

## Repository Structure

```text
Week_exercises/
├── week1/
├── week2/
├── .gitignore
└── README.md
```

More weekly exercises will be added as the course progresses.

## Course

**IN4080 – Natural Language Processing**
University of Oslo (UiO)
Autumn 2026

## Author

**Zejing Wang**

Master's student in Mathematics for Applications
University of Oslo
