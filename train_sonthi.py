import os
import warnings
warnings.filterwarnings("ignore")

from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
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
    output_path = "./"

    dataset_config = BaseDatasetConfig(
        name="ljspeech",
        language='th',
        meta_file_train="/root/sonthi_train_data/metadata.csv",
        path=os.path.join(output_path, "/root/sonthi_train_data"),
    )

    character_config = CharactersConfig(
        # characters_class= 'Graphemes',
        pad='<PAD>',
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        punctuations="!'(),-.:;? ",
        characters='กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลวศษสหฬอฮะัาำิีึืุูเแโใไ็่้๊๋์​‎'
    )

    audio_config = BaseAudioConfig(
        sample_rate=16000,
        do_trim_silence=True,
        trim_db=60.0,
        signal_norm=False,
        mel_fmin=0.0,
        mel_fmax=8000,
        spec_gain=1.0,
        log_func="np.log",
        ref_level_db=20,
        preemphasis=0.0,
    )

    config = Tacotron2Config(
        audio=audio_config,
        batch_size=32,
        eval_batch_size=16,
        num_loader_workers=25,
        num_eval_loader_workers=5,
        run_eval=False,
        test_delay_epochs=-1,
        ga_alpha=0.0,
        decoder_loss_alpha=0.25,
        postnet_loss_alpha=0.25,
        postnet_diff_spec_alpha=0,
        decoder_diff_spec_alpha=0,
        decoder_ssim_alpha=0,
        postnet_ssim_alpha=0,
        r=2,
        test_sentences=['เรา อาจ จะ ลอง พูด ดู ก่อน',
                        'เคราะห์หามยามร้าย กาลเทศะ',
                        'ความ ไม่ มี โรค เป็น ลาภ อัน ประเสริฐ'],
        characters=character_config,
        attention_type="dynamic_convolution",
        double_decoder_consistency=False,
        epochs=10000,
        text_cleaner="basic_cleaners",
        use_phonemes=False,
        phoneme_language="th",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=50,
        print_eval=False,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
    )

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        formatter=formatter,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    model = Tacotron2(config, ap, tokenizer)
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )
    trainer.fit()
