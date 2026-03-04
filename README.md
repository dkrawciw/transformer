# Mines AI

Sawyer Allen

Shane Ritter

Daniel Krawciw

We based our LLM on various Ancient Greek texts (English translations), with texts from philosphers, historians, playwrights and poets. We pulled these from Project Gutenberg and the list can be found in the bestmodelconfig.txt. Since we are training this model on a laptop, the outputs are not very good and the most coherent parts of outputs are only a few words long.

## Instructions

### Step 1.
Run uv sync to ensure the packages are correctly installed.
```bash
uv sync
```

### Step 2
We have a model that is pre-trained with better hyperparameters in data/saved_model.pkl. If you don't wish to use it, delete the file and the model will be retrained on a smaller dataset with smaller hyperparameters. If you want it back, just move best_model_stuff/best_saved_model.pkl to data/ and rename it to saved_model.pkl



