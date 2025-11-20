import tkinter as tk
from tkinter.messagebox import showinfo
from tkinter.ttk import Button

from src.speech_processor import SpeechProcessor

recording_started = False

sp = SpeechProcessor(model_name="tiny")
sp.initialize_whisper_model()

def start_recording(button: Button, text_field: tk.Text):
    global recording_started

    if not recording_started:
        recording_started = True
        button["text"] = 'Идёт запись ..'

        text_field.delete("1.0", tk.END)

        val: bool = sp.detect_speech()
        if val:
            showinfo(title="Информация", message="Текст готов, остановите запись")

    else:
        recording_started = False
        button["text"] = 'Начать запись'

        sp.save_to_wav()
        text = sp.transcribe()
        text_field.insert("1.0", text)


root = tk.Tk()
root.geometry("500x400")
root.title("Speech Translator")

btn = tk.Button(
    text="Начать запись",
    font=("Arial", 12)
)
output_text = tk.Text(
    root,
    height=15,
    width=50,
    font=("Arial", 12)
)

btn.config(command=lambda: start_recording(btn, output_text))

btn.pack(pady=20)
output_text.pack(padx=20, pady=10)

root.mainloop()
