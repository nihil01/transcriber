import os
from queue import Queue
from tempfile import mkstemp

import whisper as wh
import speech_recognition as sr


class SpeechProcessor:

    def __init__(self, model_name):
        self.model: wh.Whisper = None
        self.text = ""
        self.wav_name = ""

        #запись будет в указанный период времени
        self.default_duration_time = 10
        self.microphone = sr.Microphone(sample_rate=16000)

        self.data_queue = Queue()
        self.recorder = sr.Recognizer()

        if model_name != "tiny":
            raise ValueError("Используйте tiny")
        self.model_name = model_name

    def initialize_whisper_model(self):
        print(f"Инициализируется модель '{self.model_name}' ...")
        self.model = wh.load_model(
            name=self.model_name,
            device="cpu"
        )

    def detect_speech(self):

        # создаем реальный временный файл
        fd, path = mkstemp(suffix=".wav")
        os.close(fd)
        self.wav_name = path
        self.text = ""

        # калибровка шума
        with self.microphone as source:
            print("Активировал микрофон")
            #атоматическое завершение записи ( по уровню энергии )
            # self.recorder.adjust_for_ambient_noise(source)

            #получение аудиопотока за предоставленный промежуток времени
            audio = self.recorder.record(source, duration=self.default_duration_time)
            print("Данные")
            print(audio)
            self.data_queue.put(audio)
            return True


    def save_to_wav(self):
        #Сохранение данных; Из очереди сначала читается AudioData, далее из нее извлекается wav_data
        # и это пишется во временный файл и всё
        chunks = b""
        while not self.data_queue.empty():
            audio: sr.AudioData = self.data_queue.get()
            chunks += audio.get_raw_data()

        if not chunks:
            print("Нет данных")
            return

        audio_data = sr.AudioData(chunks, sample_rate=16000, sample_width=2)
        wav_bytes = audio_data.get_wav_data()

        with open(self.wav_name, "wb") as f:
            f.write(wav_bytes)

        print(f"WAV был сохранён в {self.wav_name}")


    def transcribe(self) -> str:
        #здесь транскрибируется файл в текст с указанным языком. Так как модель самая лёгкая (tiny) -
        # то полученный текст зачастую неточен, можно использовать другие доступные модели для точности - они просто тяжелее

        if not self.model:
            return "Модель не инициализирована"

        options = {
            "language": "ru",
            "task": "transcribe"
        }

        audio = wh.load_audio(self.wav_name)

        #тут и происходит преобразование в текст
        segments = wh.transcribe(self.model, audio, **options, fp16=False)

        #удалить временный файл записи
        try:
            os.unlink(self.wav_name)

        except FileNotFoundError:
            print("Файл не найден")

        return segments["text"]
