from src.MinesAI import MinesAI
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

ai_untrained = MinesAI(
            #aristotle politics, plato republic, aristotle ethics, plato symposium, plato apology, iliad, odyssey, greek tragedies
            #aristophanes lysistrata, herodotus histories x2, xenophon anabasis, xenophon hellenica, athenian constitution, history of pelo. war
            gutenberg_ids = [6762],
            d_model = 128, 
            d_hidden = 4*128,
            d_head = 32,
            n_context = 64,
            n_layers = 2,
            donttrain=True
        )


ai_trained = MinesAI.load_model()

prompt = "The greatest Greek city is:"
length = 10
temp = 0.7
print(f"Prompt is {prompt}")
print("Untrained model:")
print(prompt+" "+ai_untrained.generate_text(prompt, length, temp))
print("Trained Model:")
print(prompt+" "+ai_trained.generate_text(prompt, length, temp))