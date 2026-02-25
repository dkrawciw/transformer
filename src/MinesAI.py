import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Our own classes/functions
from Transformer import Config, Transformer
from tokenizer import (
    gutenberg_to_tokenized,
    tokenized_to_vocab,
    encode,
    decode,
)

class MinesAI:
    """
    # MinesAI

    This is our class to initialize, train, and prompt our very own transformer AI.

    *Note:* This model is only trained on data from Project Guttenberg
    """

    def __init__(self, 
                 gutenberg_ids: list[int],
                 d_model: int,
                 d_hidden: int,
                 n_context: int,
                 n_layers: int,
    ):
        # Grab the data from Project Gutenberg
        self.training_data, self.tokenized_training_data = gutenberg_to_tokenized(gutenberg_ids)
        self.vocab_arr, self.vocab_dict = tokenized_to_vocab(self.tokenized_training_data)

        # Set all of the configurations based off of parameters and word dictionary from Project Gutenberg list
        self.config = Config(
            d_model = d_model, 
            d_vocab = len(self.vocab_dict), 
            d_hidden = d_hidden, 
            n_context = n_context, 
            n_layers = n_layers,
        )

        # Instantiate the model object to be trained
        self.model = Transformer(self.config)

        # Variables for training the model
        self.data_encoded = encode(self.training_data, self.vocab_dict)
        self.training_data = torch.utils.data.TensorDataset(torch.from_numpy(self.data_encoded[:-1]),torch.from_numpy(self.data_encoded[1:]))
        self.training_loader = torch.utils.data.DataLoader(self.training_data, batch_size=4, shuffle=True)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # train the model on the data
        self.__train_model()

    def __train_one_epoch(self, epoch_index: int, tb_writer: SummaryWriter):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in tqdm(enumerate(self.training_loader), desc="Training model over batches of data", colour="blue"):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(self.training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    def __train_model(self, num_epochs: int = 1):
        epoch_number = 0
        best_vloss = 1_000_000.

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        
        for i, epoch in tqdm(enumerate(range(num_epochs)), desc="Training Data over Epochs"):
            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.__train_one_epoch(epoch_number, writer)

            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch_number + 1)
            writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1

def main():
    ai = MinesAI(
        gutenberg_ids = [6762, 1497, 8438, 1600, 1656],
        d_model = 10,
        d_hidden = 15,
        n_context = 20,
        n_layers = 2,
    )