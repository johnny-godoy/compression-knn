# Compression KNN Classifier

## Introduction

This is a text classifier based on KNN algorithm. It is a simple and easy to use.
It's implemented with scikit-learn interface, using vectorized operations and 
caching for fast performance, with minimal dependencies.
It's based on simple text compression algorithm, which is used to calculate the distance between two texts.
By default, it uses the familiar `gzip` compressor.

It can even be used for non-text tasks, by simply converting the data to text.

## Usage
You may install it with pip:

```bash
pip install git+https://github.com/johnny-godoy/compression-knn.git
```

We implement the scikit-learn interface, so it can be used like other scikit-learn classifiers.

```python
from compression_knn.knn import CompressionKNNClassifier

X_train = [
    "red, round, sweet",
    "orange, round, tangy",
    "red, oblong, sweet",
    "orange, oblong, tangy",
    "green, round, sour"
]
y_train = ["Apple", "Orange", "Apple", "Orange", "Apple"]
X_test = ["yellow, round, sweet", "green, round, sweet"]


clf = CompressionKNNClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)

# Output:
# ['Apple', 'Apple']
```

## Upcoming
* Implementation of `CompressionKNNClassifierCV` for fast hyperparameter tuning
* Classification performance comparison notebooks
* Implementation of a vector-to-text scikit-learn compatible transformer for non-text 
  tasks

These will be gradually implemented in the `dev` branch. Once all functionality is done, version 1.0.0 will release!

## References

[“Low-Resource” Text Classification: A Parameter-Free Classification Method with 
Compressors](https://aclanthology.org/2023.findings-acl.426) (Jiang et al., Findings 2023)
