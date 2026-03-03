import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dataclasses import asdict
import traceback
import matplotlib.pyplot as plt

# Our own classes/functions
from Transformer import Config, Transformer
from tokenizer import (
    gutenberg_to_tokenized,
    tokenized_to_vocab,
    encode,
    decode,
)

# Create a directory for trained object saves
import pickle as pkl
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

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
                 d_head: int,
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
            d_head=d_head,
            n_context = n_context, 
            n_layers = n_layers,
        )


        # Instantiate the model object to be trained
        self.model = Transformer(self.config)

        # Variables for training the model
        self.data_encoded = encode(self.training_data, self.vocab_dict)
        self.training_data = torch.utils.data.TensorDataset(torch.from_numpy(self.data_encoded[:-1]),torch.from_numpy(self.data_encoded[1:]))
        self.training_loader = torch.utils.data.DataLoader(self.training_data, batch_size=self.config.n_context, shuffle=True)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, nesterov=True)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_List = []

        # train the model on the data
        self.__train_model()

        self.save_model()
    
    
    def getModelLoss(self):
        return self.loss_List

    def __train_one_epoch(self, epoch_index: int, tb_writer: SummaryWriter):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in tqdm(enumerate(self.training_loader), desc="Training model over batches of data", colour="blue"):
            # Every data instance is an input + label pair

            
            if i == len(self.training_loader)-1:
                break

            inputs, labels = data

            # Zero your gradients for every batch
            self.optimizer.zero_grad()
            

            try:
                outputs = self.model(inputs)
            except Exception as e:
                print("Training failed.")
                print("Config:")
                for name, value in asdict(self.config).items():
                    print(f"  {name} = {value}")
                print(len(inputs))
                traceback.print_exc()
                raise

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                self.loss_List.append(last_loss)
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

    def generate_text(self, text: str, max_length: int, temp: float):
        #print(self.vocab_dict)
        original_text = text
        text_tokenized = torch.from_numpy(encode(text=text.lower(), vocab_dict=self.vocab_dict))
        diff_from_n_context = text_tokenized.shape[0] - self.config.n_context
        if diff_from_n_context > 0:
            text_tokenized = text_tokenized[diff_from_n_context: ]
        elif diff_from_n_context < 0:
            text_tokenized = torch.nn.functional.pad(text_tokenized, (-1*diff_from_n_context, 0))
        self.model.eval()
        

        for i in range(max_length):
            text_tokenized = text_tokenized[-self.config.n_context:] 

            with torch.no_grad():
                output = self.model(text_tokenized)

            probs = torch.softmax(output[-1]/temp, dim=0)
            output_token = torch.multinomial(probs, num_samples=1)
            #output_token = torch.argmax(output[-1], dim=0).unsqueeze(0)
            text_tokenized = torch.cat([text_tokenized, output_token])
        
        new_tokens = text_tokenized[-max_length:]
        new_text = decode(new_tokens.numpy(), vocab_arr=self.vocab_arr)

        return original_text + " " + new_text
    
    def save_model(self) -> None:
        with open(DATA_DIR / "saved_model.pkl", "wb") as pkl_file:
            pkl.dump(self, pkl_file)
    
    @staticmethod
    def load_model():
        assert (DATA_DIR / "saved_model.pkl").exists()

        with open(DATA_DIR / "saved_model.pkl", "rb") as pkl_file:
            ai = pkl.load(pkl_file)
        
        return ai

def main():
    if (DATA_DIR / "saved_model.pkl").exists():
        ai = MinesAI.load_model()
        # save a loss curve
        train_losses = ai.getModelLoss()
        batches = []
        for i in range(1, len(train_losses)+1):
            batches.append(i * 1000)


        # Plotting
        plt.plot(batches, train_losses, label='Training Loss')
        plt.title('Training Loss Curve')
        plt.xlabel('Batchs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig("LossCurve.jpg", format='jpeg', dpi=100)
        print(ai.generate_text("Plato's favorite color is", 5, 0.75))
    else:
        ai = MinesAI(
            #aristotle politics, plato republic, aristotle ethics, plato symposium, plato apology, iliad, odyssey, greek tragedies
            #aristophanes lysistrata, herodotus histories x2, xenophon anabasis, xenophon hellenica, athenian constitution, history of pelo. war
            gutenberg_ids = [6762, 1497, 8438, 1600, 1656, 2199, 1727, 7073, 7700, 2707, 2456, 1170, 1174, 26095, 7142],
            #gutenberg_ids= [6762],
            d_model = 64, 
            d_hidden = 4*64,
            d_head = 16,
            n_context = 32,
            n_layers = 2,
        )

        # save a loss curve
        train_losses = ai.getModelLoss()
        batches = []
        for i in range(1, len(train_losses)+1):
            batches.append(i * 1000)


        # Plotting
        plt.plot(batches, train_losses, label='Training Loss')
        plt.title('Training Loss Curve')
        plt.xlabel('Batchs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig("LossCurve.jpg", format='jpeg', dpi=100)
    
        print(ai.generate_text("Plato's favorite color is", 5, 0.5))


if __name__ == "__main__":
    main()