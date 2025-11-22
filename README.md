[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/KWGyG2CI)
HW11 Problem 6: MPL Capacity Experiment
=======================================

Introduction
------------

Neural networks are universal function approximators. As we increase the
number of hidden units, a network gains the **capacity** to model more
complicated functions. However, larger models can also memorize the
training data and perform poorly on unseen examples. Fully‑connected
networks have a trade‑off which can be explored by training networks of
different sizes on a synthetic two‑dimensional classification task.

In this assignment you will reproduce a similar capacity experiment. You
will train a multilayer perceptron (MLP) with one hidden layer of
varying size using scikit‑learn's `MLPClassifier` with the L‑BFGS
optimizer. You will measure both the **cross‑entropy loss** (a.k.a. log
loss) and the **classification error rate** on the training and test
splits. Finally, you will plot how performance changes as you increase
the hidden‑layer size.

Dataset
-------

To keep the exercise self‑contained we use a small toy dataset generated
with scikit‑learn's `make_moons` function. It produces two interleaving
half circles in the plane. We draw 200 examples for training and 200
examples for testing with a little noise added to make the problem
non‑trivial. Each example consists of a 2‑D input `x` and a binary label
`y∈{0,1}`.

The notebook will provide code to generate these variables:

    from sklearn.datasets import make_moons

    # Training set
    x_tr_N2, y_tr_N = make_moons(n_samples=200, noise=0.2, random_state=0)

    # Test set
    x_te_T2, y_te_T = make_moons(n_samples=200, noise=0.2, random_state=1)

Objectives
----------

1.  **Construct MLP models of different sizes.** For each hidden size in
    `size_list`` = [4, 16, 64]` you will build an MLP classifier with:

2.  One hidden layer of the specified size (use a list of size one when
    passing to `hidden_layer_sizes`).

3.  ReLU activation: `activation='``relu``'`.

4.  L‑BFGS solver: `solver='``lbfgs``'`.

5.  A maximum of 1000 iterations: `max_iter``=1000`.

6.  A deterministic random seed (use `random_state``=``run_id` where
    `run_id` is the index of the run, starting at zero). In this
    simplified problem we only do **one run per size** (`n_runs``=1`).

7.  **Train and evaluate.** Fit the classifier on the training data
    `(x_tr_N2, ``y_tr_N``)` and then obtain class probabilities on both
    training and test sets using `predict_proba``()`. Compute the
    cross‑entropy loss using `sklearn.metrics.log_loss` and the error
    rate using `sklearn.metrics.zero_one_loss`. Store the results in
    two‑dimensional arrays `tr_loss_arr`, `te_loss_arr`, `tr_err_arr`
    and `te_err_arr` of shape `(S, ``n_runs``)`, where `S` is the number
    of hidden sizes.

8.  **Plot your results.** Using Matplotlib, make two plots. The first
    should show training and test loss versus hidden size; the second
    should show training and test error versus hidden size. A simple
    line plot with distinct markers for train and test is sufficient.
    Remember to label your axes and add a legend.

9.  **Interpretation (optional).** Consider how the model's performance
    changes as you increase the hidden‑layer size. Does the training
    loss always decrease? What happens to the test loss and error?
    Relate your observations to the concept of model capacity and
    overfitting.

Implementation tips
-------------------

-   **One run per size.** To keep this exercise light, set
    `n_runs`` = 1`. You can always experiment with multiple seeds later.

-   **Use arrays to store results.** Initialize four arrays of shape
    `(``len``(``size_list``), ``n_runs``)` filled with zeros. Within the
    loops assign each element of the arrays with the corresponding
    metric.

-   **Check your imports.** You will need `numpy`, `matplotlib.pyplot`,
    `MLPClassifier` from `sklearn.neural_network` and the metrics from
    `sklearn.metrics`. Feel free to import additional modules such as
    `time` if you wish to measure runtime, although timing is not
    required here.

-   **Commented solution.** In the provided notebook the solution code
    is supplied but commented out. Replace the `TODO` lines and
    uncomment the provided solution when you are ready to run your
    experiment. If you leave the placeholders in place, the code will
    raise a `NotImplementedError` as a reminder to finish your
    implementation.

By following these steps you will reproduce a classic capacity
experiment and gain intuition about how the size of a network's hidden
layer affects its ability to fit data and generalize to new examples.
