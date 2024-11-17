import os
import gzip
import re
import logging
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def read_and_clean_file(file_path):

    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return []

    logging.info(f"Original number of lines in {file_path}: {len(lines)}")

    cleaned_lines = []
    for line in lines:
        line = line.strip()  # 去除行首尾空白
        if line:
            # 替换多个空白字符为单个空格
            line = re.sub(r'\s+', ' ', line)
            # 去除所有非汉字、非字母、非数字字符（保留基本标点符号）
            line = re.sub(r'[^\w\s\u4e00-\u9fa5，。！？、；：]', '', line)
            # 去除多余的空格
            line = line.strip()
            cleaned_lines.append(line)

    logging.info(f"Cleaned number of lines in {file_path}: {len(cleaned_lines)}")

    return cleaned_lines

def main():

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    files = [f for f in os.listdir('.') if f.endswith('2017.zh.shuffled.deduped.gz')]
    all_sentences = []

    for file in files:
        file_path = os.path.join('.', file)
        sentences = read_and_clean_file(file_path)
        all_sentences.extend(sentences)

    train_data, val_data = train_test_split(all_sentences, test_size=1000, random_state=42)

    logging.info(f"Len of Train Set: {len(train_data)}")
    logging.info(f"Len of Val Set: {len(val_data)}")

    try:
        with open('train_data_2017.txt', 'w', encoding='utf-8') as f:
            for sentence in tqdm(train_data, desc="Writing train data"):
                f.write(sentence + '\n')

        with open('val_data_2017.txt', 'w', encoding='utf-8') as f:
            for sentence in tqdm(val_data, desc="Writing val data"):
                f.write(sentence + '\n')
    except Exception as e:
        logging.error(f"Error writing data to file: {e}")

if __name__ == "__main__":
    main()


