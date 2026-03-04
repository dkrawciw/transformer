# Mines AI

Sawyer Allen

Shane Ritter

Daniel Krawciw

We based our LLM on various Ancient Greek texts (English translations), with texts from philosphers, historians, playwrights and poets. We pulled these from Project Gutenberg and the list can be found in the bestmodelconfig.txt. Since we are training this model on a laptop, the outputs are not very good and the most coherent parts of outputs are only a few words long.

## Instructions

### Step 1: Sync packages
Run uv sync to ensure the packages are correctly installed.
```bash
uv sync
source .venv/bin/activate
```

### Step 2: Run the chatbot
Run our main file to talk to "Plato and Aristotle" (they aren't as smart as their real-life counterparts)!

```bash
uv run main.py
```

In main.py, you can change the max_length variable to change the length of the output, and you can change the temp variable to control the temperature of the model.

### Step 3: Rerun the model modifying the parameters

We have a pretrained model in `data/saved_model.pkl`. If you wish to retrain the model, **delete** the file, and run `uv run main.py`. The model will be retrained on a smaller dataset with smaller hyperparameters. If you want it back, just move `best_model_stuff/best_saved_model.pkl` to `data/` and rename it to `saved_model.pkl`.

## File Structure

`src/` - Contains actual code for the transformer and chatbot

`data/` - Stores trained models to be loaded

`best_model_stuff/` - Stores the model with the most training and our loss curve

`test/` - Test files to give easy gut-checks to know if everything is working correctly