import os
import codecs
import pandas as pd
from pythainlp.transliterate import transliterate
from pythainlp.transliterate import romanize
from tqdm import tqdm

if __name__ == '__main__':
    print(transliterate("ฉลอง ความ สำเร็จ ที่ สยาม", engine="ipa"))
    # print(transliterate("ไป ไหน กัน", engine="ipa"))
    # print(transliterate("แบบ นี้ ไม่ ดี หรอก", engine="ipa"))
    #
    # print(transliterate("จอห์นสันคอนโทรลส์อินเตอร์เนชั่นแนล", engine="tltk_ipa"))
    #
    # print(romanize("เรา จะ ไป ไหน กัน ดี นะ", engine="thai2rom"))
    #
    # print(romanize("ณคุณ", engine="thai2rom"))
    # print(romanize("ที่ ผ่าน มา ผม เชื่อ ว่า เรา สามารถ ที่ จะ ไป ได้ ไกล กว่า นี้", engine="thai2rom"))
    # print(romanize("ที่ผ่านมาผมเชื่อว่าเราไปได้ไกลกว่านี้", engine="tltk"))
    # print(romanize("อาจจะไม่ถึงขั้นที่ว่าดีมากแต่แค่พอใช้ได้ก็พอแล้ว", engine="tltk"))
    #
    # # print(transliterate("โอเค ไหม", engine="tltk_ipa"))



    df = pd.read_csv('/Users/Nakhun/sonthi_manifest.csv',
                     encoding='utf-8', names=['file_path', 'text'])
    file = codecs.open("metadata_clean.csv", "a", "utf-8")
    file_ = codecs.open("metadata_clean_all_ch.csv", "a", "utf-8")

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        file_name = os.path.basename(row['file_path']).replace('.wav', '')
        text = row['text'].strip()
        text = text.replace('ดี.ซี.', 'ดี ซี')
        text = text.replace('คศ.', 'คอ ศอ')
        text = text.replace('ปอท.', 'ปอ ตอ ทอ')
        text = text.replace('\u200b', '')
        text = text.replace('\u200e', '')
        text = text.replace('\u200e', '')
        text = transliterate(text, engine="ipa")
        file_ .write('{}'.format(text))

        file.write('{}|{}|{}\n'.format(file_name, text, text))

    fh = open('/Users/Nakhun/Projects/vits/src/metadata_clean_all_ch.csv', 'r').read()
    unique_chars = set(fh)
    print(unique_chars)
    len(unique_chars)  # for the length

    chars = ['d', '.', 'ɯ', 'ʔ', 'l', 'ɕ', 'ɛ', 'j', 'u', ' ', 'm', 'r', 'f', 'h', 'ʉ', '̯', 'ɤ', 'i', 'ː', 'k', 's', 'n', 'a', 'ə', 'ŋ', 't', 'o', 'w', 'p', '͡', 'b', 'e', 'ʰ', 'ɔ']
    all_chars = ''
    for char in unique_chars:
        all_chars = f'{all_chars}{char}'
    print(all_chars)





