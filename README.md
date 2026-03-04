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

# Write Up

## Results

Are results were much better on the trained data then the untrained. For the best model trained we have a total net loss of 22. With it starting around 28 ending around 6. When the model was untrained we got the same word over and over. Our trained model gets a somewhat cohesive sentence, but it still doesn't really understand the context of the surrounding text, however it still does significantly better than our untrained model.

### Example outputs

Untrained model:

The greatest Greek city is: city city city city city city city city city city

Trained model:

The greatest Greek city is: of the braves , he will be and end ,

## Design Choices

We went with an OOP transformer with some tests in the test dir/folder. Then to not run a training loop multiple times we saved the model into a .pkl file. This was done in the MinesAI.py file. We chose this as it would result in a more organized code base. Which would be easier to run and find where code pieces were.

## Challenges

The first main challenge was implementing the transformer itself. Mainly the attention head. This mainly included looking up syntax and torch modules as we had little experience using pytorch. It also included making sure we had our dimensions right throughout the transformer. This turned out to be a problem later in the implementation once we got to our training loop as we had to go update the transformer class to make sure the transformers matrixes were the right dimensions.

Another problem we had was the implementation of the Embedding as it would not get the right dimension. We had a d_vocab x 1 matrix and we were struggling to figure out how to make it dn x dm.

The last issue we faced was with the masking matrix as it didn't hardcode it so we had to make sure the masking matrix was nc x nc, when we ran the generate function.

## Future Directions

Some future directions could include better embedding as well as improving tokenization/using a different type of tokenization. We could also train more data on the model which would also improve the LLM. We could also refine our hyperparameters a little more.

# Contributions

Shane Ritter: Helped find syntax for pytorch. Added the transformer class which ran through are TBs and embedding. Coded up the start of the train() and the __train_one_epoch(). Created the plot for the loss curve. Wrote most of the write up.

Sawyer Allen:

Daniel Krawciw: Split code into files, setup tests, compiled gutenberg and tokenize code from Michael's work for our purposes, general debugging, and pkl file code.