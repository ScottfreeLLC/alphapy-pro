Command Line Interface
======================

The AlphaPy Pro Command Line Interface (CLI) provides a streamlined way
to run machine learning pipelines. Simply navigate to your project
directory and run ``alphapy``.

Basic Usage
-----------

First, change to your project directory::

    cd path/to/project

Train models and generate predictions::

    alphapy

Core Commands
-------------

**alphapy** - Main ML Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Usage::

    alphapy [options]

Options:

* ``--train`` - Train new models and make predictions [Default]
* ``--predict`` - Score the test data with a trained run (the most recent, or ``--rundir``)
* ``--rundir`` - The run to score with: a path, or a run name such as ``run_20261115_210812`` under ``runs/``
* ``--debug`` - Enable debug mode with detailed output

Example::

    # Train models (default behavior)
    alphapy

    # Score data/test with the most recent run
    alphapy --predict

    # Score with a chosen run
    alphapy --predict --rundir run_20261115_210812

Predict mode scores every row of ``data/test`` (labelled or not) with every algorithm of the run and the blend, with
the feature steps as fitted on the run's training rows, and writes ``output/ranked_predict`` in the format of
training's ``ranked_test``. Training's own files are kept. Window algorithms (the ``sequence`` section) read each
group's last training rows as history when the group's rows come after them, as training's test rows did; otherwise
the file must carry its own earlier rows. Predict mode refuses to run with ``live_results`` set, because a
live-results step reading the newest ``output/ranked_test*`` would publish training's test predictions.

From Python, ``alphapy.inference.load_run(run_dir).predict_proba(frame)`` returns each algorithm's and the blend's
probabilities for the rows of a frame (the raw columns training loaded), indexed like the frame. For window
algorithms, the frame carries enough earlier rows per group (``window - 1``), or ``history=`` supplies them; a row
short of history is refused unless ``allow_short_history=True``. Runs trained before the
feature pipeline was saved (``model/feature_pipeline.pkl``) cannot be scored this way; train them again.

**Custom Pipelines**
~~~~~~~~~~~~~~~~~~~~

You can create custom data-preparation pipelines around AlphaPy by producing
the canonical ``train``/``test`` inputs expected by the package and then
running ``alphapy`` from the project directory.

Project Structure Requirements
------------------------------

Before running commands, ensure your project follows this structure::

    my_project/
    ├── config/
    │   └── model.yml       # Model configuration
    ├── data/
    │   ├── train.csv       # Training data
    │   └── test.csv        # Testing data (optional)
    └── runs/               # Output directory (auto-created)

Output Structure
----------------

After running a command, outputs are organized in timestamped directories::

    runs/
    └── run_YYYYMMDD_HHMMSS/
        ├── config/         # Configuration used
        ├── input/          # Input data snapshots
        ├── model/          # Trained models and metrics
        ├── output/         # Predictions and rankings
        └── plots/          # Visualizations

Common Workflows
----------------

**1. Quick Model Training**::

    cd projects/my_project
    alphapy

**2. Hyperparameter Tuning**:

First, edit ``config/model.yml`` to enable grid search::

    model:
        grid_search:
            option: True
            iterations: 100

Then run::

    alphapy --verbose

**3. Batch Predictions**::

    # Train once
    alphapy --train
    
    # Use the model for multiple prediction sets
    alphapy --predict

Logging and Debugging
---------------------

AlphaPy Pro creates detailed logs:

* ``alphapy.log`` - Main pipeline log

To increase log verbosity::

    alphapy --verbose --debug

Tips and Best Practices
-----------------------

1. **Always verify your data** before running models
2. **Start with small iterations** for grid search, then increase
3. **Use dated runs** for reproducibility
4. **Keep model.yml under version control**
5. **Review plots** in the output directory for insights

For more details on configuration options, see :doc:`../user_guide/project`.
