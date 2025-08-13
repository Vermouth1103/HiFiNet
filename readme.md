# HiFiNet Official Repository

This repository contains the official code of **HiFiNet**.

---

## Environment Setup

Please create the Python environment as specified in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate hifinet
```

## Representation Model Training

To train the HiFiNet representation model, run like

```python
python main_exp/beijing/train.py
```

## Downstream Task Model Training

After training the representation model, you can train the downstream task model (e.g., label prediction) by running:

```python
python main_exp/beijing/downstream_task/train_label_prediction.py
```

## Notes

- Make sure all dependencies are installed via environment.yml.

- Adjust any paths or configuration files in the scripts as needed for your environment.

- This code is intended for research purposes.

