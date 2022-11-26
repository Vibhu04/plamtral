from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import torch
#from create_dataset import generate_dataset


class TL_Dataset(Dataset):

    def __init__(self, base_model, model_size, dataset_path, block_size):
        """
        Reads data from a txt file and chunks it into args.block_size batches.
        Feel free to modify this file as per your requirements.
        """

        super().__init__()
        #generate_dataset(MAX_LINE_LEN)
        if base_model == 'GPT2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2' + model_size)
        else:
            raise Exception("Error: this library supports only GPT2 as the base model for now.")
        self.dataset_path = dataset_path
        self.sentences = []
        self.text = ''

        with open(dataset_path) as file:

            for line in file:
                self.text += line.strip('\n')

            tokenized_text = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.text))

            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                self.sentences.append(torch.tensor(self.tokenizer.build_inputs_with_special_tokens(tokenized_text[i: i + block_size])))


    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        return self.sentences[item]