from transformers import AutoTokenizer
from torch.utils.data import Dataset
import pickle
import os

class DatasetTokenized(Dataset):
    def __init__(self, data_path, n=3):
        """
        n: n-gram的窗口大小
        """
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
        self.n = n
        self.data, self.labels = self.load_data(data_path)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx], self.labels[idx]

    def load_data(self, path):

        data = []
        labels = []

        try:
            with open(path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for line in lines:
                    tokenized = self.tokenizer(line.strip(), return_tensors='pt', truncation=True)['input_ids'][0]
                    # print(tokenized)
                    # Create n-grams
                    if len(tokenized) >= self.n:
                        for i in range(len(tokenized) - self.n + 1):
                            # Append the n-1 tokens to the data list
                            data.append(tokenized[i:i + self.n - 1])
                            # print(data)
                            # Append the nth token to the labels list
                            labels.append(tokenized[i + self.n - 1])
                            # print(labels)
        except Exception as e:
            print(f"Error loading data: {e}")
        # print(data, labels)
        return data, labels

class DatasetSentence(Dataset):

    def __init__(self, data_path):
        self.data = self.get_data(data_path)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]

    def get_data(self, path):

        sentence = []
        try:
            with open(path, 'r', encoding='gbk', errors='ignore') as file:
                data = file.readlines()
                for sample in data:
                    sentence.append(sample.strip())
            print('Data loaded successfully!')
        except FileNotFoundError:
            print(f"Error: The file {path} was not found.")
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")

        return sentence

if __name__ == '__main__':
    data_path = 'test.txt'
    dataset = DatasetTokenized(data_path, n=3)
    print(len(dataset))
    print(dataset[0])

    dataset = DatasetSentence(data_path)
    print(len(dataset))
    print(dataset[0])

