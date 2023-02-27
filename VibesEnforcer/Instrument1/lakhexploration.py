
import mido
import os
import joblib
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')
# enable svg
%config InlineBackend.figure_format = 'svg'

import tqdm

filesDir = '/Users/cameronfranz/Desktop/composerAssistant/lmd_full/c'
filesList = os.listdir(filesDir)[:10000]

def processFile(file):
  try:
    midiFile = mido.MidiFile(filesDir + '/' + file)
    return midiFile
  except Exception as e:
    return None

midiFiles = joblib.Parallel(n_jobs=8, verbose=1)(joblib.delayed(processFile)(file) for file in tqdm.tqdm(filesList))
midiFiles = [f for f in midiFiles if f is not None]

def getTempo(midiFile):
  try: 
    for msg in midiFile:
      if msg.type == 'set_tempo':
        tempoRaw = msg.tempo
        tempo = mido.tempo2bpm(tempoRaw)
        return tempo
  except Exception as e:
    pass
  return None

# with ProcessPoolExecutor(max_workers=8) as executor:
tempos = joblib.Parallel(n_jobs=8, verbose=1)(joblib.delayed(getTempo)(midiFile) for midiFile in tqdm.tqdm(midiFiles))
formatted = [t for t in tempos if t is not None]
plt.hist(formatted, bins=100)
plt.show()

def isSoloPiano(midiFile):
  pianoPrograms = [0,1,2,3,4,5,6,7]
  try:
    for msg in midiFile:
      if msg.type == 'program_change' and msg.program not in pianoPrograms:
        return False
    return True
  except Exception as e:
    pass
  return False

isSoloPianoList = joblib.Parallel(n_jobs=8, verbose=1)(joblib.delayed(isSoloPiano)(midiFile) for midiFile in tqdm.tqdm(midiFiles))

soloPianoTempos = np.array(tempos)[np.array(isSoloPianoList)]
formattedTempos = [t for t in soloPianoTempos.tolist() if t is not None]
plt.hist(formattedTempos, bins=100)
plt.show()
mode = max(set(formattedTempos), key=formattedTempos.count)
mode

