import sherpa_onnx, wave, numpy as np
d="sherpa-onnx-whisper-small/"
r=sherpa_onnx.OfflineRecognizer.from_whisper(encoder=d+"small-encoder.int8.onnx",decoder=d+"small-decoder.int8.onnx",tokens=d+"small-tokens.txt",language="ar",task="transcribe",num_threads=3)
w=wave.open("audio.wav"); x=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)/32768
for a,b in [(0.54,2.95),(3.64,6.78),(7.71,9.3),(9.7,10.88),(11.49,13.69),(14.77,15.16)]:
    s=r.create_stream(); s.accept_waveform(16000,np.concatenate([np.zeros(4000,np.float32),x[int((a-.1)*16000):int((b+.1)*16000)]])); r.decode_stream(s); print(a,b,s.result.text,flush=True)
