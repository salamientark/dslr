# DSLR

A data science project focused on building a multi-class classifier using logistic regression (One-vs-Rest, OvR). The goal is to analyze and process data, then implement a custom algorithm for accurate classification.
## Training Process

- Select features and target from the dataset.
- Clean the data (missing values are replaced with feature means).
- Standardize features for consistent scaling.
- Set initial parameters (learning rate, iterations, weights).
- Train the classifier using logistic regression for each target class (OvR).
- Unstandardize weights to return to the original scale.

## Model Testing

**Parameters:**
- Learning rate: 0.1
- Iterations: 100

**Feature Sets & Accuracy:**
- Astronomy, Herbology, Ancient Runes  
  Accuracy: 98.19%
- Astronomy, Herbology, Ancient Runes, Defense Against the Dark Arts  
  Accuracy: 98.19%
- Astronomy, Herbology, Ancient Runes, Divination  
  Accuracy: 98.19%
- Astronomy, Herbology, Ancient Runes, Divination, Charms  
  Accuracy: 98.19%
- Astronomy, Herbology, Ancient Runes, Flying  
  Accuracy: 98.19%


### Way for improvment
Change row by row calculation by matrix calculation
