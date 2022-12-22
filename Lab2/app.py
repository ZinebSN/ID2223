from pytube import YouTube
from transformers import pipeline
import gradio as gr
import os

pipe = pipeline(model="Yilin98/whisper-small-hi")  # change to "your-username/the-name-you-picked"

def get_audio(url):
  yt = YouTube(url)
  stream = yt.streams.filter(only_audio=True).first()
  out_file=stream.download(output_path=".")
  base, ext = os.path.splitext(out_file)
  new_file = base+'.mp3'
  os.rename(out_file, new_file)
  audio = new_file
  return audio


def transcribe(audio=None, file=None, youtube=None):
    if (audio is None) and (file is None) and (youtube is None):
        return "No audio provided!"
    elif audio is not None:
        input=audio
    elif file is not None:
        input=file
    elif youtube is not None:
        input=get_audio(youtube)
    text = pipe(input)["text"]
    return text

iface = gr.Interface(
    fn=transcribe, 
    inputs=[
        gr.Audio(source="microphone", type="filepath", interactive=True),
        gr.Audio(source="upload", type="filepath", interactive=True),
        gr.Text(label="URL (YouTube, etc.)")], 
    outputs="text",
    title="Whisper Small Swedish",
    description="Realtime demo for Swedish speech recognition using a fine-tuned Whisper small model.",
)

iface.launch()