import numpy as np
from jaxtyping import Int
import requests
import unicodedata
from collections import Counter
from pathlib import Path
from tqdm import tqdm

def get_gutenberg_book(
	id: int | None = 84,
	data_temp: Path | str = "../data/gutenberg_data",
	remove_gutenberg_meta: bool = True,
) -> str:
	"""
	# Get Gutenberg Book
	
	This method gets the text of a particular book from Project Gutenberg
	"""
	
	data_temp: Path = Path(data_temp)
	data_temp.mkdir(parents=True, exist_ok=True)
	
	url: str = f"https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
	data_path: Path = Path(data_temp) / f"{id}.txt"
	data: str
	# read from cache if it exists
	if data_path.exists():
		with open(data_path, 'r', encoding='utf-8') as file:
			data = file.read()
	else:
		# download if it doesn't exist
		response: requests.Response = requests.get(url)
		response.raise_for_status()  # Ensure that the download was successful
		data = response.text

		# save to cache
		with open(data_path, 'w', encoding='utf-8') as file:
			file.write(data)

	# remove header/footer
	if remove_gutenberg_meta:
		data = '***'.join(data.split('***')[2:])
		data = '***'.join(data.split('***')[:-1])
	
	return data

def get_many_books(
		ids: list[int],
		data_temp: Path | str = "../data/gutenberg_data",
	) -> list[str]:
	"""
	# Get Many Books
	
	This method will take a list of ids of Project Gutenberg books and compiles the text into one list of strings.
	
	## Parameters
	* ids: list of book ids
	* data_temp: Either a Path object or string path to the location where data can temporarily be stored
	"""
	
	data: list[str] = []
	for id in tqdm(ids, desc="Getting Guttenberg Book Text"):
		item: str = get_gutenberg_book(id, data_temp)
		data.append(item)
	
	return data

def process_text(
	text: str,
	allowed_punctuation: str = "-.,;:!?()\"\\" + "".join(str(x) for x in range(10)),
	punctuation_convert: dict[str, str] = {'—': '-'},
) -> str:
	
	# replace some special characters which unicode won't normalize properly
	for char, replacement in punctuation_convert.items():
		text = text.replace(char, replacement)

	# if a line has ".jpg" in it, remove that line (this is specific to Don Quixote)
	text = '\n'.join(
		line 
		for line in text.split('\n')
		if '.jpg' not in line
	)

	# Normalize the string to decompose Unicode characters
	text = unicodedata.normalize('NFKD', text)

	# Encode to ASCII bytes, then decode back to string, ignoring errors
	text = text.encode('ascii', 'ignore').decode('ascii')

	# remove newlines and tabs
	text = text.replace('\n', ' ').replace('\t', ' ')


	# put spaces around allowed punctuation
	for char in allowed_punctuation:
		text = text.replace(char, f' {char} ')


	# remove leading and trailing spaces
	text = text.strip()

	# remove multiple spaces
	while '  ' in text:
		text = text.replace('  ', ' ')

	# remove all characters except (alphanumeric, allowed_punctuation, ' ')
	text = ''.join(
		(
			char 
			if (
				char.isalnum() 
				or char in allowed_punctuation 
				or char == ' '
			)
			else ' '
		)
		for char in text 
	)

	# convert to lowercase
	text = text.lower()

	text = text.strip()

	return text

def tokenize(
	text: str,
	process: bool = False,
) -> list[str]:
	"""
	# Tokenize
	"""
    
	if process:
		text = process_text(text)
	return text.split(' ')

def encode(
	text: str | list[str],
	vocab_dict: dict[str,int],
) -> Int[np.ndarray, " n_tokens"]:
	if isinstance(text, str):
		text = tokenize(text)
	return np.array([vocab_dict[word] for word in text])

def decode(
	encoded_text: Int[np.ndarray, " n_tokens"] | list[int],
	vocab_arr: list[str],
) -> str:
	return ' '.join(vocab_arr[i] for i in encoded_text)

def gutenberg_to_tokenized(
	ids: list[int],
) -> tuple[str, list[str]]:
	
	data_raw = get_many_books(ids)
	data = " ".join(process_text(x) for x in data_raw)
	data_tokenized = tokenize(data)
	
	return data, data_tokenized

def tokenized_to_vocab(
	data_tokenized: list[str],
) -> tuple[list[str], dict[str, int]]:
	
    vocab_freq: Counter[str] = Counter(data_tokenized)
    vocab_arr: list[str] = [word for word, _ in vocab_freq.most_common()]
    vocab_dict: dict[str, int] = {word: i for i, word in enumerate(vocab_arr)}

    return vocab_arr, vocab_dict

def main():
	"""Example of tokenizing plato and aristotle and encoding and decoding the data"""
    # Getting books from Plato and Aristotle
	data, data_tokenized = gutenberg_to_tokenized([6762, 1497, 8438, 1600, 1656])
	vocab_arr, vocab_dict = tokenized_to_vocab(data_tokenized)
	
    # Encoding and Decoding the gutenberg data
	data_encoded: Int[np.ndarray, " n_tokens"] = encode(data, vocab_dict)
	data_decoded: str = decode(data_encoded, vocab_arr)
	
	print(f"Original:\n{data[:20]}\n")
	print(f"Encoded:\n{data_encoded[:20]}\n")
	print(f"Decoded:\n{data_decoded[:20]}\n")


if __name__ == "__main__":
	main()