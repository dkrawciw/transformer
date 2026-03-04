## Results


# Example outputs


## Design Choices

We went with an OOP transformer with some tests in the test dir/folder. Then to not run a training loop multiple times we saved the model into a .pkl file. This was done in the MinesAI.py file. We chose this as it would result in a more organized code base. Which would be easier to run and find where code pieces were.

## Challenges

Are first main challenge was implementing the transformer itself. Mainly the attention head. This mainly included looking up syntax and torch modules as we had little experience using pytorch. It also included making sure we had are dimensions right throughout the transformer. This turned out to be a problem latter in the implementation once we got to are training loop as we had to go update the transformer class to make sure are transformers matrixes were the right dimensions.


Another problem we had was the implementation of the Embedding as it we would not get the right dimensions as we had d_vocab x 1 matrix and we were struggling to figure out how to make it dn x dm.


The last issue we faced was with are masking matrix as it was didn't hardcode it so we had to make sure the masking matrix was nc x nc, when we ran are generate function.


## Future Directions
Some future directions could include better embedding as well as improving are tokenization/using a different type of tokenization. We could also train more data on the model which would also improve the LLM. We could also refine are hyperparameters a little more.


