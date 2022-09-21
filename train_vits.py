import os
import warnings

warnings.filterwarnings("ignore")

from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


def formatter(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "sonthi"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            text = cols[1]
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name})
    return items


if __name__ == '__main__':
    output_path = os.path.dirname(os.path.abspath(__file__))
    dataset_config = BaseDatasetConfig(
        name="ljspeech",
        meta_file_train="/root/sonthi_train_data/metadata_clean.csv",
        path="/root/sonthi_train_data")

    audio_config = VitsAudioConfig(
        sample_rate=16000,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None
    )

    char_config = CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters="pa̯nkɯŋihlɔʉɕʔbtəmj͡wousrʰɛdeɤːf",
        punctuations=". ",
        phonemes=None
    )
    config = VitsConfig(
        audio=audio_config,
        run_name="vits_ljspeech",
        batch_size=32,
        eval_batch_size=16,
        batch_group_size=5,
        num_loader_workers=20,
        num_eval_loader_workers=16,
        run_eval=False,
        test_delay_epochs=-1,
        epochs=10000,
        text_cleaner="basic_cleaners",
        use_phonemes=False,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        compute_input_seq_cache=True,
        print_step=50,
        print_eval=False,
        mixed_precision=True,
        use_language_weighted_sampler=True,
        output_path=output_path,
        datasets=[dataset_config],
        characters=char_config,
        cudnn_benchmark=False,
        test_sentences=['tʰiː pʰaːna maː pʰama t͡ɕʰɯːa̯ waː reːaː saːmaːntʰa',
                        't͡ɕa paj daj kajla kwaː niː',
                        't͡ɕʰalɔːŋa kʰwaːma samreːt͡ɕa tʰiː sajaːm'])

    ap = AudioProcessor.init_from_config(config)

    tokenizer, config = TTSTokenizer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        formatter=formatter,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size)

    model = Vits(config, ap, tokenizer, speaker_manager=None)

    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples)

    trainer.fit()
