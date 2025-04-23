# Linear Regression

1. What are the pros and cons of using the normal equation to solve for the weights in linear regression as opposed to using gradient descent?


1. Pros and Cons of Using the Normal Equation vs. Gradient Descent
The normal equation is a closed-form solution for finding the optimal weights in linear regression:




Pros of the Normal Equation
1. Exact Solution: Unlike gradient descent, which iteratively updates weights, the normal equation provides an exact solution in a single computation.
2. No Hyperparameters: Gradient descent requires choosing a learning rate (?\alpha?) and setting stopping criteria, while the normal equation does not.
3. Fast for Small Datasets: For small datasets (low feature count), matrix inversion is computationally feasible and provides a quick solution.
Cons of the Normal Equation
1. Computational Complexity: The normal equation requires computing (X^T X)^{-1}, which has a complexity of O(n^3). This is impractical for large feature sets.
2. Singular Matrix Issue: If X^T X is non-invertible (due to multicollinearity or redundant features), the normal equation cannot be computed directly.
3. Memory Intensive: The matrix inversion operation requires storing an n×n matrix in memory, making it inefficient for very high-dimensional data.
When to Use Each Method
* Use the Normal Equation for small datasets (few features).
* Use Gradient Descent when dealing with large datasets or high-dimensional data, as it scales better and does not require matrix inversion.



# Logistic Regression

1. Why is the softmax function used in multi-class logistic regression (Hint: the model itself produces logits)?

1. Why is the Softmax Function Used in Multi-Class Logistic Regression?
In binary logistic regression, the sigmoid function is used to map raw model outputs (logits) into probabilities:



However, for multi-class classification, the softmax function is used instead:



where zk? is the raw logit (score) for class k, and K is the total number of classes.
Why Use Softmax?
1. Probability Distribution: The softmax function ensures that the output values sum to 1, making them interpretable as probabilities.
2. Handles Multiple Classes: Unlike sigmoid (which is for binary classification), softmax assigns a probability to each class, allowing multi-class classification.
3. Logits to Probabilities: Since logistic regression produces logits (raw scores), softmax converts them into meaningful class probabilities.
4. Compatible with Cross-Entropy Loss: The softmax function pairs well with cross-entropy loss, which optimizes the model for correct classification.
Key Takeaway
* Sigmoid is used for binary classification (outputs a probability between 0 and 1).
* Softmax is used for multi-class classification (outputs a probability distribution across classes).

