import os
from tqdm import tqdm
import pandas as pd
from pythainlp.transliterate import transliterate

if __name__ == '__main__':
    output_dir = '/home/poctest/nakhun/tts/vits/filelists'
    df = pd.read_csv('/home/da/Documents/data/experiment_setup/tts/sonthi_manifest.csv',
                     encoding='utf-8', names=['file_path', 'text'])
    row_count = len(df)
    df = df.sample(frac=1).reset_index(drop=True)

    train_df = df.head(row_count - 2000)
    valid_df = df.tail(2000)

    raw_output_path = os.path.join(output_dir, 'train_raw.txt')
    clean_output_path = os.path.join(output_dir, 'train_clean.txt')

    raw = open(raw_output_path, 'a', 'utf-8')
    clean = open(clean_output_path, 'a', 'utf-8')

    for index, row in tqdm(train_df.iterrows(), total=train_df.shape[0], desc='Processing Train Set'):
        file_name = os.path.basename(row['file_path'])
        file_path = os.path.join('/home/da/Documents/data/experiment_setup/tts', file_name)
        res = transliterate(row['text'], engine="ipa")

        clean.write(f'{file_path}|{res}\n')
        raw.write(f'{file_path}|{row["file_path"]}\n')

    raw.close()
    clean.close()

    raw_output_path = os.path.join(output_dir, 'valid_raw.txt')
    clean_output_path = os.path.join(output_dir, 'valid_clean.txt')

    raw = open(raw_output_path, 'a', 'utf-8')
    clean = open(clean_output_path, 'a', 'utf-8')

    for index, row in tqdm(valid_df.iterrows(), total=valid_df.shape[0], desc='Processing Validation Set'):
        file_name = os.path.basename(row['file_path'])
        file_path = os.path.join('/home/da/Documents/data/experiment_setup/tts', file_name)
        res = transliterate(row['text'], engine="ipa")

        clean.write(f'{file_path}|{res}\n')
        raw.write(f'{file_path}|{row["file_path"]}\n')

    raw.close()
    clean.close()
    