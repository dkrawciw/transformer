import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from tokenizer import (
    process_text,
    tokenize,
    encode,
    decode,
    tokenized_to_vocab,
    get_gutenberg_book,
    get_many_books,
)


class TestProcessText:
    def test_basic_text(self):
        result = process_text("Hello World")
        assert result == "hello world"

    def test_punctuation_spacing(self):
        result = process_text("Hello, world!")
        assert " , " in result and " !" in result

    def test_unicode_normalization(self):
        result = process_text("café")
        assert "cafe" in result or result == "caf "

    def test_special_char_conversion(self):
        result = process_text("em—dash")
        assert "em - dash" in result

    def test_multiple_spaces_removed(self):
        result = process_text("hello    world")
        assert "  " not in result

    def test_newlines_and_tabs(self):
        result = process_text("hello\nworld\there")
        assert "\n" not in result and "\t" not in result

    def test_mixed_case_and_punctuation(self):
        result = process_text("Hello, World! (Test)")
        assert result.islower()


class TestTokenize:
    def test_basic_tokenization(self):
        result = tokenize("hello world test")
        assert result == ["hello", "world", "test"]

    def test_empty_string(self):
        result = tokenize("")
        assert result == [""]

    def test_tokenize_with_processing(self):
        result = tokenize("Hello, World!", process=True)
        assert all(token.islower() or token in "-.,;:!?()\"\\" for token in result)

    def test_single_token(self):
        result = tokenize("hello")
        assert result == ["hello"]


class TestEncodeAndDecode:
    def test_encode_string(self):
        vocab_dict = {"hello": 0, "world": 1}
        result = encode("hello world", vocab_dict)
        assert np.array_equal(result, np.array([0, 1]))

    def test_encode_list(self):
        vocab_dict = {"hello": 0, "world": 1}
        result = encode(["hello", "world"], vocab_dict)
        assert np.array_equal(result, np.array([0, 1]))

    def test_decode(self):
        vocab_arr = ["hello", "world"]
        encoded = np.array([0, 1])
        result = decode(encoded, vocab_arr)
        assert result == "hello world"

    def test_decode_with_list(self):
        vocab_arr = ["hello", "world"]
        result = decode([0, 1], vocab_arr)
        assert result == "hello world"

    def test_encode_decode_roundtrip(self):
        vocab_dict = {"test": 0, "data": 1, "here": 2}
        vocab_arr = ["test", "data", "here"]
        original = "test data here"
        encoded = encode(original, vocab_dict)
        decoded = decode(encoded, vocab_arr)
        assert decoded == original


class TestTokenizedToVocab:
    def test_vocab_creation(self):
        data = ["hello", "world", "hello", "test"]
        vocab_arr, vocab_dict = tokenized_to_vocab(data)
        assert "hello" in vocab_arr
        assert "world" in vocab_arr
        assert vocab_dict["hello"] == vocab_arr.index("hello")

    def test_vocab_frequency_ordering(self):
        data = ["a", "a", "a", "b", "b", "c"]
        vocab_arr, vocab_dict = tokenized_to_vocab(data)
        assert vocab_arr[0] == "a"  # Most frequent first

    def test_vocab_dict_consistency(self):
        data = ["x", "y", "z"]
        vocab_arr, vocab_dict = tokenized_to_vocab(data)
        for word, idx in vocab_dict.items():
            assert vocab_arr[idx] == word


@patch('tokenizer.requests.get')
def test_get_gutenberg_book(mock_get, tmp_path):
    mock_response = MagicMock()
    mock_response.text = "***\n***\nBook Content\n***\n***"
    mock_get.return_value = mock_response

    result = get_gutenberg_book(id=123, data_temp_raw=tmp_path, remove_gutenberg_meta=True)
    assert "Book Content" in result


@patch('tokenizer.get_gutenberg_book')
def test_get_many_books(mock_get_book):
    mock_get_book.side_effect = ["Book 1", "Book 2"]
    result = get_many_books([1, 2], data_temp="../data")
    assert len(result) == 2
    assert result == ["Book 1", "Book 2"]


class TestProcessTextEdgeCases:
    def test_only_punctuation(self):
        result = process_text(".,!?")
        assert result.strip()  # Should have content after processing

    def test_digits_preserved(self):
        result = process_text("test123")
        assert "1 2 3" in result

    def test_jpg_line_removal(self):
        result = process_text("line one\nimage.jpg content\nline two")
        assert "image.jpg" not in result
        assert "line one" in result

    def test_empty_string_input(self):
        result = process_text("")
        assert result == ""

    def test_whitespace_only(self):
        result = process_text("   \n\t  ")
        assert result == ""


class TestTokenizeEdgeCases:
    def test_tokenize_with_punctuation(self):
        result = tokenize("hello, world!")
        assert "hello," in result or "," in result

    def test_tokenize_preserves_order(self):
        text = "first second third fourth"
        result = tokenize(text)
        assert result.index("first") < result.index("second")


class TestEncodeEdgeCases:
    def test_encode_with_unknown_word_raises_error(self):
        vocab_dict = {"hello": 0}
        with pytest.raises(KeyError):
            encode("hello unknown", vocab_dict)

    def test_encode_empty_list(self):
        vocab_dict = {"hello": 0}
        result = encode([], vocab_dict)
        assert len(result) == 0

    def test_decode_empty_array(self):
        vocab_arr = ["hello", "world"]
        result = decode([], vocab_arr)
        assert result == ""


class TestTokenizedToVocabEdgeCases:
    def test_single_word_vocab(self):
        data = ["hello", "hello", "hello"]
        vocab_arr, vocab_dict = tokenized_to_vocab(data)
        assert len(vocab_arr) == 1
        assert vocab_dict["hello"] == 0

    def test_empty_tokenized_data(self):
        data = []
        vocab_arr, vocab_dict = tokenized_to_vocab(data)
        assert len(vocab_arr) == 0
        assert len(vocab_dict) == 0


@patch('tokenizer.requests.get')
def test_get_gutenberg_book_caching(mock_get, tmp_path):
    mock_response = MagicMock()
    mock_response.text = "***\n***\nCached Content\n***\n***"
    mock_get.return_value = mock_response

    result1 = get_gutenberg_book(id=456, data_temp_raw=tmp_path, remove_gutenberg_meta=True)
    result2 = get_gutenberg_book(id=456, data_temp_raw=tmp_path, remove_gutenberg_meta=True)
    
    assert result1 == result2
    assert mock_get.call_count == 1  # Should only download once


@patch('tokenizer.requests.get')
def test_get_gutenberg_book_no_meta_removal(mock_get, tmp_path):
    mock_response = MagicMock()
    mock_response.text = "***\n***\nContent\n***\n***"
    mock_get.return_value = mock_response

    result = get_gutenberg_book(id=789, data_temp_raw=tmp_path, remove_gutenberg_meta=False)
    assert "***" in result