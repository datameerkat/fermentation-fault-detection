# fermentation-fault-detection

This repository is part of my master's thesis "Framework for data-driven fault detection models in fermentation processes". The thesis presents the development and application of a framework for creating datadriven fault detection models using synthetic data.

## Installation

To set up the environment and install the project as a package, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/fermentation-fault-detection.git
   cd fermentation-fault-detection
   ```
2. **Install the project as a package**
    This install the package in "editable" mode, making it easier to adjust the code.
    ```bash
    pip install -e .
    ```

## Usage instructions

It is recommend to follow the workflow of the framework presented in the thesis. A short overview is given here.

1. **Generated datasets**
    Datasets (training, validation and test) need to be generated with the "make_dataset.ipynb" notebook. If reproduction of results is the goal, further information about the datasets can be found in the thesis document.

2. **Optimize ML models**
    Based on the generated datasets, machine learning based fault detection models can be trained with the notebooks found in the "Optimize_ML_models" folder. Make sure to adjust the names of the training and validation dataset according to the names of the datasets you created.

3. **Evaluate the model performance**
    After the fault detection models were optimized and saved, their performance can be evaluated with the "model_evaluation.ipynb" notebook. For this, a separate test dataset should be used.

4. **(Optional) Evaluated extrapolation capabilities**
    The generalization capabilities of the models can be evaluated using the "extrapolation_comparision.ipynb" notebook. This requires the models to evaluated on two different test sets, where one test set is based on parameter distributions diverging from the ones of the training set.

## Project structure

The directory structure of the project looks like this:

```txt
├── LICENSE
├── Makefile
├── README.md
├── data
│   ├── Monte_Carlo             <- stores parameter distributions of datasets
│   └── simulation_sets         <- generated datasets
├── evaluation                  <- stores model evaluation results
├── fermfaultdetect             <- source code of the project
│   ├── data                    <- code to generate or analyse datasets
│   │   ├── make_dataset.py
│   │   ├── preprocessing.py
│   │   ├── statistics.py
│   │   └── utils.py
│   ├── fault_detect_models     <- ML model classes for fault detection
│   │   └── ml_models.py
│   ├── model_evaluation.py     <- code for evaluation of models
│   ├── process_models
│   │   └── fed_batch_model.py  <- process model used to generate data
│   ├── utils.py
│   └── visualizations
│       └── visualize.py        <- code for visualisations
├── models                      <- optimized fault detection & diagnosis models developed in the thesis
│   ├── ANN_detect_thesis
│   ├── ANN_diagnosis_FE_thesis
│   ├── ANN_diagnosis_thesis
│   ├── MPCA_Thesis_highalpha
│   ├── MPCA_Thesis_lowalpha
│   ├── PLSDA_Thesis
│   └── SVM_Thesis
├── notebooks
│   ├── Noise analysis          <- determine airflow and weight noise
│   ├── Optimize_ML_models      <- scripts to optimize and save fault detection/diagnosis models
│   │   ├── ANN_detect.ipynb
│   │   ├── ANN_diagnosis.ipynb
│   │   ├── MPCA.ipynb
│   │   ├── PLS_fdm.ipynb
│   │   └── SVM_fdm.ipynb
│   ├── extrapolation_comparison.ipynb  <- compares evaluation results
│   ├── make_dataset.ipynb              <- generate datasets
│   ├── model_evaluation.ipynb          <- evaluated model performance
│   └── visualize_dataset.ipynb         <- visualize features of a dataset
├── pyproject.toml                      <- project configuration file
├── requirements.txt                    <- requirements for reproducing the environment
├── Thesis.pdf                          <- Thesis document
```

Initially created using [mlops_template](https://github.com/SkafteNicki/mlops_template).
