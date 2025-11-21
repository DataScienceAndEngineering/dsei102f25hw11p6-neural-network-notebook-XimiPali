"""
Unit tests for the simplified capacity experiment notebook.

These tests are designed to be used with pytest within a GitHub Classroom
autograder.  They read the Jupyter notebook, check that all TODOs have
been filled in, execute the notebook in a clean namespace and verify
that an MLP classifier with the correct configuration has been used
and that the resulting metrics make sense.  Tests are written to
produce informative assertions rather than crashing outright.
"""

import nbformat
import numpy as np
import types

NOTEBOOK_PATH = 'hw11_P6_notebook_instructor.ipynb'

def _load_notebook(nb_path: str):
    """Load a notebook from disk."""
    with open(nb_path, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)


def _run_notebook(nb_path: str):
    """
    Execute the code cells of a notebook sequentially in a fresh
    namespace.  Returns the namespace as a dictionary.

    If a NotImplementedError is raised (because the student has
    intentionally left a TODO), this function propagates the exception
    so that the calling test can handle it gracefully.
    """
    nb = _load_notebook(nb_path)
    env: dict = {}
    # We execute each code cell in order.  Markdown cells are skipped.
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        # Skip empty or whitespace‑only code cells
        if not cell.source or cell.source.strip() == '':
            continue
        try:
            exec(cell.source, env, env)
        except NotImplementedError:
            # Re‑raise so that tests can mark missing implementations
            raise
        except Exception as e:
            # Re‑raise other exceptions so pytest will report them
            raise
    return env


def test_no_todo_left():
    """
    Ensure that there are no remaining TODO markers in any code cell.
    A student should remove or replace all TODO comments with working
    code.  Finding a TODO indicates that the implementation is
    incomplete.
    """
    nb = _load_notebook(NOTEBOOK_PATH)
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        source = cell.source
        if 'TODO' in source:
            raise AssertionError(
                f"Cell {idx} still contains a TODO. Please replace all TODOs with your implementation."
            )


def test_mlp_configuration_and_metrics():
    """
    Execute the notebook and verify the MLP configuration and result shapes.

    The test checks that:
    - The last trained model is available under the name `mlp`.
    - Its solver is 'lbfgs' and activation function is 'relu'.
    - The loss and error arrays have the correct shapes.
    - Training loss and error decrease (or stay the same) as hidden size increases.
    """
    try:
        env = _run_notebook(NOTEBOOK_PATH)
    except NotImplementedError:
        raise AssertionError(
            "Your notebook still contains unimplemented sections. Please complete all TODOs before running the tests."
        )

    # Check that mlp exists and is an instance of MLPClassifier
    mlp = env.get('mlp', None)
    assert mlp is not None, (
        "The variable `mlp` was not found in the notebook environment."
    )
    # Verify solver and activation
    assert hasattr(mlp, 'solver'), "`mlp` does not have a solver attribute."
    assert mlp.solver == 'lbfgs', f"Expected solver='lbfgs', got solver='{mlp.solver}'"
    assert hasattr(mlp, 'activation'), "`mlp` does not have an activation attribute."
    assert mlp.activation == 'relu', f"Expected activation='relu', got activation='{mlp.activation}'"

    # Extract arrays and size_list
    size_list = env.get('size_list')
    assert size_list is not None and isinstance(size_list, (list, tuple)), (
        "`size_list` must be defined as a list or tuple of hidden sizes."
    )
    n_runs = env.get('n_runs')
    S = len(size_list)
    tr_loss_arr = env.get('tr_loss_arr')
    te_loss_arr = env.get('te_loss_arr')
    tr_err_arr  = env.get('tr_err_arr')
    te_err_arr  = env.get('te_err_arr')
    # Check shapes
    for name, arr in [('tr_loss_arr', tr_loss_arr), ('te_loss_arr', te_loss_arr),
                      ('tr_err_arr', tr_err_arr), ('te_err_arr', te_err_arr)]:
        assert isinstance(arr, np.ndarray), f"{name} should be a numpy array."
        assert arr.shape == (S, n_runs), (
            f"{name} should have shape ({S}, {n_runs}), got {arr.shape}."
        )
    # Check monotonic decrease (non‑increasing) for training metrics over sizes
    tr_losses = tr_loss_arr[:, 0].tolist()
    tr_errs   = tr_err_arr[:, 0].tolist()
    for i in range(len(tr_losses) - 1):
        assert tr_losses[i+1] <= tr_losses[i] + 1e-6, (
            "Training loss should not increase as the hidden layer size increases. "
            f"Got {tr_losses[i]} -> {tr_losses[i+1]}."
        )
        assert tr_errs[i+1] <= tr_errs[i] + 1e-6, (
            "Training error should not increase as the hidden layer size increases. "
            f"Got {tr_errs[i]} -> {tr_errs[i+1]}."
        )
