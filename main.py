from src.MinesAI import MinesAI
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

def main():
    if (DATA_DIR / "saved_model.pkl").exists():
        ai = MinesAI.load_model()
    else:
        ai = MinesAI(
            #gutenberg_ids = [6762, 1497, 8438, 1600, 1656],
            gutenberg_ids= [6762],
            d_model = 10, 
            d_hidden = 15,
            d_head = 2,
            n_context = 20,
            n_layers = 10,
        )

    max_length = 20
    print("Ask Plato and Aristotle Bot anything you want!\nEnter \"q!\" to quit")

    while(True):
        print()
        user_input = input("> ")

        if user_input == ("q!"):
            break

        ai_resp = ai.generate_text(f"{user_input} Plato and Aristotle say: ", max_length)
        print(ai_resp)
    
    rand_word = str(ai.generate_text("Plato and Aristotle", max_length=1))
    
    print(f"Plato and Aristotle {rand_word} goodbye!")

if __name__ == "__main__":
    main()