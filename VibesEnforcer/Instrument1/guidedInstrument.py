# netsh interface portproxy set v4tov4 listenport=8888 listenaddress=0.0.0.0 connectport=8888 connectaddress="172.27.72.197"
#  netsh interface portproxy set v4tov4 listenport=8889 listenaddress=0.0.0.0 connectport=8889 connectaddress="172.27.72.197"

import os
import pickle
import random
import secrets
import statistics
from time import time
import tqdm
import torch
import matplotlib.pyplot as plt
from torchsummary import summary
from sklearn import metrics
from midi2audio import FluidSynth
from IPython.display import Audio, display
%cd ./Los-Angeles-Music-Composer
from lwa_transformer import *
import TMIDIX

full_path_to_model_checkpoint = "/home/cameronfranz/Los_Angeles_Music_Composer_Trained_Model_66010_steps_0.7282_loss.pth" #@param {type:"string"}
SEQ_LEN = 4096
model = LocalTransformer(
    num_tokens = 2831,
    dim = 1024,
    depth = 24,
    causal = True,
    local_attn_window_size = 512,
    max_seq_len = SEQ_LEN
).cuda()
model.load_state_dict(torch.load(full_path_to_model_checkpoint))
model.eval()
# summary(model)

full_path_to_model2_checkpoint = "/home/cameronfranz/Los_Angeles_Music_Composer_Model_88835_steps_0.643_loss.pth"
model2 = LocalTransformer(
    num_tokens = 2831,
    dim = 1024,
    depth = 36,
    causal = True,
    local_attn_window_size = 512,
    max_seq_len = SEQ_LEN
)
model2 = torch.nn.DataParallel(model2).cuda()
model2.load_state_dict(torch.load(full_path_to_model2_checkpoint))
model2.eval()



def scoreToTokens(score, model2=False, num_instr_control=None, include_start_header=True):
  events_matrix = []
  melody_chords_f = []
  melody_chords_f1 = []
  itrack = 1
  patches = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

  patch_map = [[0, 1, 2, 3, 4, 5, 6, 7], # Piano
            [24, 25, 26, 27, 28, 29, 30], # Guitar
            [32, 33, 34, 35, 36, 37, 38, 39], # Bass
            [40, 41], # Violin
            [42, 43], # Cello
            [46], # Harp
            [56, 57, 58, 59, 60], # Trumpet
            [71, 72], # Clarinet
            [73, 74, 75], # Flute
            [-1], # Drums
            [52, 53], # Choir
            [16, 17, 18, 19, 20] # Organ
            ]

  while itrack < len(score):
    for event in score[itrack]:
        if event[0] == 'note' or event[0] == 'patch_change':
            events_matrix.append(event)
    itrack += 1

  events_matrix.sort(key=lambda x: x[1])

  events_matrix1 = []

  for event in events_matrix:
    if event[0] == 'patch_change':
        patches[event[2]] = event[3]

    if event[0] == 'note':
        event.extend([patches[event[3]]])
        once = False

        for p in patch_map:
            if event[6] in p and event[3] != 9: # Except the drums
                event[3] = patch_map.index(p)
                once = True

        if not once and event[3] != 9: # Except the drums
            event[3] = 15 # All other instruments/patches channel
            event[5] = max(80, event[5])

        if event[3] < 12: # We won't write chans 12-16 for now...
            events_matrix1.append(event)
            # stats[event[3]] += 1

  #=======================================================
  # PRE-PROCESSING

  # checking number of instruments in a composition
  instruments_list_without_drums = list(set([y[3] for y in events_matrix1 if y[3] != 9]))

  if len(events_matrix1) > 0 and len(instruments_list_without_drums) > 0:
    # recalculating timings
    for e in events_matrix1:
      if model2 == False:
        e[1] = math.ceil(e[1] / 8) # Max 1 seconds for start-times
        e[2] = math.ceil(e[2] / 16) # Max 2 seconds for durations
      elif model2 == True:
        e[1] = int(e[1] / 10) # Max 1 seconds for start-times
        e[2] = int(e[2] / 20) # Max 2 seconds for durations


    # Sorting by pitch, then by start-time
    events_matrix1.sort(key=lambda x: x[4], reverse=True)
    events_matrix1.sort(key=lambda x: x[1])

    #=======================================================
    # FINAL PRE-PROCESSING

    melody_chords = []

    pe = events_matrix1[0]

    for e in events_matrix1:
      if e[1] >= 0 and e[2] > 0:

        # Cliping all values...
        tim = max(0, min(127, e[1]-pe[1]))
        dur = max(1, min(127, e[2]))
        cha = max(0, min(11, e[3]))
        ptc = max(1, min(127, e[4]))
        vel = max(8, min(127, e[5]))

        velocity = round(vel / 15)

        # Writing final note
        melody_chords.append([tim, dur, cha, ptc, velocity])

        pe = e

    if len([y for y in melody_chords if y[2] != 9]) > 12: # Filtering out tiny/bad MIDIs...

      times = [y[0] for y in melody_chords[12:]]
      avg_time = sum(times) / len(times)

      times_list = list(set(times))

      instruments_list = list(set([y[2] for y in melody_chords]))
      num_instr = len(instruments_list)
      if num_instr_control is not None:
        num_instr = num_instr_control


      if True or avg_time < 112 and instruments_list != [9]: # Filtering out bad MIDIs...
        if True or 0 in times_list: # Filtering out (mono) melodies MIDIs (i.e. no chords)

            #=======================================================
            # FINAL PROCESSING
            #=======================================================

            # Break between compositions / Intro seq

            if 9 in instruments_list:
              drums_present = 2818 # Yes
            else:
              drums_present = 2817 # No

            melody_chords_f.extend([2816, drums_present, 2819+(num_instr-1)])

            #=======================================================

            # Composition control seq
            if model2 == False:
              intro_mode_time = statistics.mode([y[0] for y in melody_chords if y[2] != 9])
            elif model2 == True:
              intro_mode_time = statistics.mode([0] + [y[0] for y in melody_chords if y[2] != 9 and y[0] != 0])
            intro_mode_dur = statistics.mode([y[1] for y in melody_chords if y[2] != 9])
            intro_mode_pitch = statistics.mode([y[3] for y in melody_chords if y[2] != 9])
            intro_mode_velocity = statistics.mode([y[4] for y in melody_chords if y[2] != 9])

            # Instrument value 12 is reserved for composition control seq
            intro_dur_vel = (intro_mode_dur * 8) + (intro_mode_velocity-1)
            intro_cha_ptc = (12 * 128) + intro_mode_pitch

            if include_start_header:
              melody_chords_f.extend([intro_mode_time, intro_dur_vel+128, intro_cha_ptc+1152])

            # TOTAL DICTIONARY SIZE 2831

            #=======================================================
            # MAIN PROCESSING CYCLE
            #=======================================================

            for m in melody_chords:

              # WRITING EACH NOTE HERE
              dur_vel = (m[1] * 8) + (m[4]-1)
              cha_ptc = (m[2] * 128) + m[3]

              melody_chords_f.extend([m[0], dur_vel+128, cha_ptc+1152])
              melody_chords_f1.append([m[0], dur_vel+128, cha_ptc+1152])
  return melody_chords_f

def tokensToSongFormat(tokens, model2=False):
  song = tokens
  song_f = []
  tim = 0
  dur = 0
  vel = 0
  pitch = 0
  channel = 0

  son = []
  song1 = []

  for s in song:
    if s >= 128 and s < (12*128)+1152:
      son.append(s)
    else:
      if len(son) == 3:
        song1.append(son)
      son = []
      son.append(s)

  for ss in song1:

    if model2 == False:
      tim += ss[0] * 8
      dur = ((ss[1]-128) // 8) * 16
    elif model2 == True:
      tim += ss[0] * 10
      dur = ((ss[1]-128) // 8) * 20
    vel = (((ss[1]-128) % 8)+1) * 15
    channel = (ss[2]-1152) // 128
    pitch = (ss[2]-1152) % 128

    song_f.append(['note', tim, dur, channel, pitch, vel ])
  return song_f

def previewSongFormat(score):
  fname = "LAMC_tmp"

  detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(score, output_signature = 'Los Angeles Music Composer', output_file_name=fname, track_name='Project Los Angeles', list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0], number_of_ticks_per_quarter=500)

  x, y, c = [], [], []

  colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'pink', 'orange', 'purple', 'gray', 'green', 'gold', 'silver']

  for s in score:
    x.append(s[1] / 1000)
    y.append(s[4])
    c.append(colors[s[3]])

  FluidSynth("/usr/share/sounds/sf2/FluidR3_GM.sf2", 16000).midi_to_audio(str(fname+'.mid'), str(fname+'.wav'))
  display(Audio(str(fname+'.wav'), rate=16000, autoplay=False))

  plt.figure(figsize=(14,12))
  ax=plt.axes(title=fname)
  ax.set_facecolor('white')
  # include piano note colored rows
  for i in range(12,96):
    plt.axhline(y=i, color='gray', alpha=0.1, linewidth=0.5)
  # color black keys slightly darker
  for i in range(12,96):
    if i % 12 in [1, 3, 6, 8, 10]:
      plt.axhline(y=i, color='gray', alpha=0.3, linewidth=4)
  plt.scatter(x,y, c=c)
  plt.xlabel("Time")
  plt.ylabel("Pitch")
  plt.show()

def customGenerate(model, prime, seq_len, temperature=0.8, filter_thres=0.9, top_p=1.0, min_stop_token=0, return_prime=False, verbose=True, **kwargs):

  # Takes logits of shape [num_tokens] -- no batch dim. In place filter :/
  def top_p_filter(logits, p):
    assert logits.dim() == 1
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    # print("Keeping {}%".format((1 - indices_to_remove.shape[0] / logits.shape[0]) * 100))
    # plt.hist(logits.cpu())
    # plt.show()
    logits_out = logits.clone()
    logits_out[indices_to_remove] = -float('Inf')
    return logits_out

  model.eval()
  n, device = prime.shape[1], prime.device
  out = prime

  with torch.no_grad():
    for s in range(seq_len):
      with torch.no_grad():
        logits = model.forward(out[:, -model.max_seq_len:], return_loss=False, **kwargs)

      filtered_logits = top_p_filter(logits[0, -1, :], top_p).unsqueeze(0) # add batch dim back so we're [batch, num_tokens]
      probs = F.softmax(filtered_logits / temperature, dim = -1)
      sampled = torch.multinomial(probs, 1)
      out = torch.cat((out, sampled), dim = -1)

      if verbose:
        if s % 32 == 0:
          print(s, '/', seq_len)

      if min_stop_token > 0:
        for sa in sampled:
            if sa >= min_stop_token:
              stop = True
              break
            else:
              stop = False
        if stop:
              if verbose:
                print('Model called the end of sequence at:', s, '/', seq_len)
              break

  if return_prime:
    return out[:, :]

  else:
    return out[:, n:]

# MIDI Server
import asyncio
import copy
import nest_asyncio
nest_asyncio.apply()
import websockets
import json
notesBuffer = []
currentlyPressedNotes = []
mainWebsocket = None
# server.close()
async def handler(websocket, path):
  global mainWebsocket
  mainWebsocket = websocket
  while(True):
    data = await websocket.recv()
    noteData = json.loads(data)
    notesBuffer.append(noteData)
    if noteData["midi"]["type"] == "note_on":
      currentlyPressedNotes.append(noteData["midi"]["note"])
    elif noteData["midi"]["type"] == "note_off":
      currentlyPressedNotes.remove(noteData["midi"]["note"])
server = await websockets.serve(handler, "0.0.0.0", 8889, ping_timeout=None)
task = asyncio.ensure_future(server.start_serving())
currentlyPressedNotes

# testMidi = "/home/cameronfranz/kgSongCC0.mid"
testMidi = "/home/cameronfranz/kgSongCC0_transposed.mid"
midiBytes = open(testMidi, "rb").read()
score = TMIDIX.midi2ms_score(midiBytes)
tokens = scoreToTokens(score) # score object is consumed X.X
score2 = tokensToSongFormat(tokens)
previewSongFormat(score2)

score = TMIDIX.midi2ms_score(midiBytes)
tokens = scoreToTokens(score, num_instr_control=9, include_start_header=True) # score object is consumed X.X
promptLength = 256
allowStop = True
modelInput = [tokens[:promptLength]] * 1
modelInput = torch.LongTensor(modelInput).cuda()
completion = customGenerate(model, modelInput, 256, temperature=1.0, top_p=0.99, return_prime=False, min_stop_token=(2816 if allowStop else 0), verbose=True, filter_thres=0.0)
completion = completion.cpu().numpy()
songOut = tokensToSongFormat(completion[0])
previewSongFormat(songOut)

score = TMIDIX.midi2ms_score(midiBytes)
tokensModel2 = scoreToTokens(score, model2=True, num_instr_control=9,include_start_header=True)
promptLength = 256
allowStop = True
modelInput = [tokensModel2[:promptLength]] * 1
modelInput = torch.LongTensor(modelInput).cuda()
completion = customGenerate(model2.module, modelInput, 256, temperature=0.9, top_p=0.99, return_prime=False, min_stop_token=(2816 if allowStop else 0), verbose=True, filter_thres=0.0)
completion = completion.cpu().numpy()
songOut = tokensToSongFormat(completion[0], model2=True)
previewSongFormat(songOut)


# previewSongFormat(tokensToSongFormat(tokens[:promptLength]))

# can try realtime note accompaniment (e.g. only 2 notes, low temp, automatic as playing). But might not work because needs to really be 120bpm.
# can also try only allowing notes that are within the currently pressed notes. (i.e. a sort of auto arpeggiator)
# can also try "fancy looper" -- i.e. start the llm with the phrase, (by default it'll loop), then affect what it does by (introducing other instruments / weighting chance of notes from diff instrument) or (weighting chance of notes of certain pitch)





























# 0
