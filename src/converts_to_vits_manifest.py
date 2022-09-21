import os
import codecs
import pandas as pd
from pythainlp.transliterate import transliterate
from pythainlp.transliterate import romanize
from tqdm import tqdm


def save_manifest(dataframe, save_path):
    file = codecs.open(save_path, "a", "utf-8")
    for index, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
        file_name = os.path.basename(row['file_path'])
        text = row['text'].strip()
        text = text.replace('ดี.ซี.', 'ดี ซี')
        text = text.replace('คศ.', 'คอ ศอ')
        text = text.replace('ปอท.', 'ปอ ตอ ทอ')
        text = text.replace('\u200b', '')
        text = text.replace('\u200e', '')
        text = transliterate(text, engine="ipa")
        file.write('/root/sonthi_train_data/wavs/{}|{}\n'.format(file_name, text))


if __name__ == '__main__':
    df = pd.read_csv('/Users/Nakhun/sonthi_manifest.csv',
                     encoding='utf-8', names=['file_path', 'text'])

    df.sample(frac=1)

    train_df = df.head(11000)
    valid_df = df[11000:]

    save_manifest(train_df, '/Users/Nakhun/Projects/vits/src/sonthi_train.txt')
    save_manifest(valid_df, '/Users/Nakhun/Projects/vits/src/sonthi_valid.txt')