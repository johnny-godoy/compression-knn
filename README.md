# Compression KNN Classifier

## Introduction

This is a text classifier based on KNN algorithm. It is a simple and easy to use.
It's implemented with scikit-learn interface, using vectorized operations and 
caching for fast performance, with minimal dependencies.
It's based on simple text compression algorithm, which is used to calculate the distance between two texts.
By default, it uses the familiar `gzip` compressor.

It can even be used for non-text tasks, by simply converting the data to text.

For vector-based inputs, use `VectorToTextTransformer` in a pipeline to convert
each row into a text sample before classification.

```python
import numpy as np
from sklearn.pipeline import Pipeline

from compression_knn.knn import CompressionKNNClassifier
from compression_knn.preprocessor import VectorToTextTransformer

X_train = np.array([[1, 1], [1, 2], [9, 9], [8, 9]])
y_train = np.array(["low", "low", "high", "high"])

pipeline = Pipeline(
  [
    ("vector_to_text", VectorToTextTransformer(separator=",")),
    ("classifier", CompressionKNNClassifier(n_neighbors=1)),
  ]
)

pipeline.fit(X_train, y_train)
print(pipeline.predict(np.array([[1, 0], [9, 8]])))
```

## Usage
You may install it with pip:

```bash
pip install git+https://github.com/johnny-godoy/compression-knn.git
```

We implement the scikit-learn interface, so it can be used like other scikit-learn classifiers.

```python
from compression_knn import CompressionKNNClassifier

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

## References

[“Low-Resource” Text Classification: A Parameter-Free Classification Method with 
Compressors](https://aclanthology.org/2023.findings-acl.426) (Jiang et al., Findings 2023)