import whisper

model = whisper.load_model('small')
result = model.transcribe("audios/audio.mp3",
                          language="pt",
                          fp16=False,
                          verbose=True,
                          patience=2,
                          beam_size=5,
                          )
print(result["text"])
