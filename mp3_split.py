from pydub import AudioSegement

audio=AudioSegement.from_file('')
interval=30*1000
num_segments=len(audio)//interval
for i in range(num_segments):
    start_time=i*interval
    end_time=(i+1)*interval
    segment=audio[start_time:end_time]
    segment.export(f"segement.{str(i).zfill(2)}",format="mp3")