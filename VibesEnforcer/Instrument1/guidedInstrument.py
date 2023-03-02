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


# takes in three tokens which represent a note
def tokensToNote(tokens, model2=False):
  deltatime = 0
  duration = 0
  velocity = 0
  pitch = 0
  channel = 0

  t = tokens

  if model2 == False:
    deltatime = t[0] * 8
    duration = ((t[1]-128) // 8) * 16
  elif model2 == True:
    deltatime = t[0] * 10
    duration = ((t[1]-128) // 8) * 20
  velocity = (((t[1]-128) % 8)+1) * 15
  channel = (t[2]-1152) // 128
  pitch = (t[2]-1152) % 128

  note =['note', deltatime, duration, channel, pitch, velocity]
  return note

def tokensToSongFormat(tokens, model2=False):
  song = tokens
  song_f = []

  time_elapsed = 0

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
    note = tokensToNote(ss)
    note[1] += time_elapsed
    time_elapsed = note[1]
    song_f.append(note)
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

# Takes logits of shape [num_tokens] -- no batch dim. In place filter :/
def top_p_filter(logits, p):
  assert logits.dim() == 1
  sorted_logits, sorted_indices = torch.sort(logits, descending=True)
  cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
  # Remove tokens with cumulative probability above the threshold
  sorted_indices_to_remove = cumulative_probs > p
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

def customGenerate(model, prime, seq_len, temperature=0.8, filter_thres=0.9, top_p=1.0, min_stop_token=0, return_prime=False, verbose=True, **kwargs):
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
    pitch = noteData["midi"]["note"]
    channel = noteData["midi"]["channel"]
    if noteData["midi"]["type"] == "note_on":
      currentlyPressedNotes.append([pitch, channel])
    elif noteData["midi"]["type"] == "note_off":
      try:
        currentlyPressedNotes.remove([pitch, channel]) # we might remove them elsewhere. TODO: don't do like tis, lol
      except Exception as e:
        pass
server = await websockets.serve(handler, "0.0.0.0", 8889, ping_timeout=None)
task = asyncio.ensure_future(server.start_serving())

# testMidi = "/home/cameronfranz/kgSongCC0.mid"
testMidi = "/home/cameronfranz/kgSongCC0_transposed.mid"
midiBytes = open(testMidi, "rb").read()
score = TMIDIX.midi2ms_score(midiBytes)
tokens = scoreToTokens(score) # score object is consumed X.X
score2 = tokensToSongFormat(tokens)
previewSongFormat(score2)

NEW_SONG_TKN = 2816

# for i in range(5):
score = TMIDIX.midi2ms_score(midiBytes)
tokens = scoreToTokens(score, num_instr_control=9, include_start_header=True, model2=False) # score object is consumed X.X
promptLength = 256
allowStop = True
modelInput = [tokens[:promptLength]] * 1
modelInput = torch.LongTensor(modelInput).cuda()
completion = customGenerate(model, modelInput, 256, temperature=0.8, top_p=0.99, return_prime=False, min_stop_token=(NEW_SONG_TKN if allowStop else 0), verbose=True, filter_thres=0.0)
completion = completion.cpu().numpy()
songOut = tokensToSongFormat(completion[0], model2=False)
previewSongFormat(songOut)

# # pregenerate biases for each note
noteBiases = torch.zeros((128, 2831), requires_grad=False).cuda()
for i in range(1152, (12*128)+1152):
  pitch = (i-1152) % 128
  channel = (i-1152) // 128
  if channel == 9:
    # ignore drums
    continue
  # only bias towards piano, guitar, bass for now
  # if channel in [0, 1, 2, 5, 6, 7]: #9 is drums
    # noteBiases[pitch][i] = 1
  noteBiases[pitch][i] = 1
  # If access noteBiases[p], will get biases in all octaves
  for p in range(pitch % 12, 128, 12):
    noteBiases[p][i] = 1


torch.set_printoptions(precision=10, sci_mode=False)
torch.tensor([1.12345e-9])
# timingBiases = torch.zeros((128, 2831), requires_grad=False).cuda()
# for i in range(128):
#   timingBiases

# [0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0][6]
temperature=0.8; top_p=0.99; promptLength = 256;
# temperature=0.8; top_p=0.99; promptLength = 3;
currentDueTime = time() + 1
currentInput = torch.LongTensor([tokens[:promptLength]]).cuda()
for i in range(12800):
  onChaPitchToken = False
  onTimingToken = False
  if currentInput[0][-1] > 1152:
    onTimingToken = True
  elif (currentInput[0][-1] >= 128 and currentInput[0][-1] < 1152):
    onChaPitchToken=True

  # if len(currentlyPressedNotes) == 0 and onTimingToken:
  #   print(currentlyPressedNotes, time(), end="\r")
  #   currentDueTime = time() + 0.1
  #   await asyncio.sleep(0.01)
  #   continue

  with torch.no_grad():
    # logits = model.forward(currentInput[:, -model.max_seq_len:], return_loss=False)[:, -1, :] # remove seq dim
    logits = model.forward(currentInput[:, -512:], return_loss=False)[:, -1, :] # remove seq dim

  currentlyPressedKorg = [n[0] for n in currentlyPressedNotes if n[1] == 0]
  # allowedNotes = []
  # for i in range(10):
  #   allowedNotes += list(range(i*12, i*12+4))
  # allowedNotes = list(range(0, 128))
  # currentlyPressedKorg = allowedNotes

  currentlyPressedArturia = [n[0] for n in currentlyPressedNotes if n[1] == 1]
  NO_NOTE_REPEAT_KEY = 36# pad 1
  ENCOURAGE_CHORD_KEY = 37#pad 2
  CHANGE_SONG_KEY = 38 #pad 3
  DISCOURAGE_CHORD_KEY = 39 # pad 4

  modified_logits = logits.clone()
  if onChaPitchToken:
    # tokensToBiasIndices = torch.tensor([[]])
    tokensToBiasIndices = torch.where(torch.sum(noteBiases[currentlyPressedKorg], axis=0) != 0)
    modified_logits[0][tokensToBiasIndices] = modified_logits[0][tokensToBiasIndices] + 6

    # tokensToRemoveIndices = torch.where((torch.sum(noteBiases[currentlyPressedKorg], axis=0) == 0))
    # print(tokensToRemoveIndices[0].shape)
    # modified_logits[0][tokensToRemoveIndices] -= 1 # could slighly bias against notes unpressed by me in last 10 seconds?

  #   if NO_NOTE_REPEAT_KEY in currentlyPressedArturia:
  #     lastChaPitchToken = currentInput[0][-3]
  #     lastPitch = (lastChaPitchToken-1152) % 128
  #     print("Biasing away from repeat, last pitch was", lastPitch, str(time()), end="\r")
  #     modified_logits[0][torch.where(noteBiases[lastPitch] != 0)] = -2000
  if onTimingToken:
    modified_logits[0][NEW_SONG_TKN] = -1000
  #   if ENCOURAGE_CHORD_KEY in currentlyPressedArturia:
  #     modified_logits[0][:4] +=8 # bias towards quickly repeated onsets in hopes of getting chords
  #     # ends up encouraging other instruments to join
  #     print("Biasing towards chords" + str(time()), end="\r")
  #   if CHANGE_SONG_KEY in currentlyPressedArturia:
  #     print("Trying to change song" + str(time()), end="\r")
  #     modified_logits[0][NEW_SONG_TKN] += 10
  #   if DISCOURAGE_CHORD_KEY in currentlyPressedArturia:
  #     modified_logits[0][:6] -= 8

  filtered_logits = top_p_filter(modified_logits[0], top_p).unsqueeze(0) # add batch dim back so we're [batch, num_tokens]
  # ideas:
  # "tight button" -- encourage either perfectly same onsets, or at least one quarter note
  # "chord button" -- encourage chords
  # "instrument join" -- encourage other instruments to join in
  # "instrument solo" -- encourage other instruments to stop playing
  # "speed up / slow down"
  # sustain key -- longer note durations

  probs = F.softmax(filtered_logits / temperature, dim = -1)

  sampled = torch.multinomial(probs, 1)
  currentInput = torch.cat((currentInput, sampled), dim = -1)

  token = sampled.cpu().item()
  if len(currentlyPressedKorg) > 0:
    if onChaPitchToken:
      # print("Probabilities of biased notes are now", torch.max(probs[0][tokensToBiasIndices].cpu(), dim=0)[0], end="\r")
      pitch = (token-1152) % 128
      # print("Sampled note with pitch", pitch)
      if token in tokensToBiasIndices[0].tolist():
        print("Sampled a biased note!")
        # Remove until it is pressed again
        for p in range(pitch % 12, 128, 12):
          if [p, 0] in currentlyPressedNotes:
            print("removing", pitch, p)
            currentlyPressedNotes.remove([p, 0])
      if pitch in currentlyPressedKorg:
        # print("Sampled note with pitch", pitch, "removing")
        # currentlyPressedNotes.remove([pitch, 0])
        pass

  if token == NEW_SONG_TKN:
    print("model started new song")

  if token < 128:
    if len(currentNoteTokens) != 0:
      # got to a new timing token without processing a note
      print("Generated incomplete note")
    currentNoteTokens = [token]
  else:
    currentNoteTokens.append(token)
    if len(currentNoteTokens) == 3:
      note = tokensToNote(currentNoteTokens, model2=False)
      currentNoteTokens.clear()
      if note[3] < 12:
        # print("sending note with pitch1", (currentNoteTokens[2] - 1152) % 128, "pitch2", note[4])
        # print("\n")
        deltaTime = note[1] / 1000
        duration = note[2] / 1000
        currentDueTime += deltaTime
        onEvent = ["note_on", currentDueTime, note[3], note[4], note[5]]
        offEvent = ["note_off", currentDueTime+duration, note[3], note[4], note[5]]
        if note[3] not in [10,  6, 7, 9, 2, 1, 0]:
          print("Got note on ch", note[3])
        else:
          asyncio.get_event_loop().create_task(mainWebsocket.send(json.dumps([onEvent, offEvent])))
  # note =['note', deltatime, duration, channel, pitch, velocity]
  # if currentDueTime < time() + 0.01:
  #   print("Model couldn't keep up ")
  #   pass
  # if currentDueTime < time() + 0.2:
  #   print("note has less than 0.5 seconds buffer", currentDueTime - time(), "seconds")
  #   currentDueTime += 0.6 # todo: model isn't aware of these changes
  if currentDueTime > time() + 0.8:
    print("note has more than 0.8 seconds buffer", currentDueTime - time())
    await asyncio.sleep(0.5)
    # currentDueTime = time() + 2 # todo: model isn't aware of these changes
  await asyncio.sleep(0.0)

previewSongFormat(tokensToSongFormat(currentInput.cpu()[0][promptLength:]))
currentlyPressedNotes

# can try realtime note accompaniment (e.g. only 2 notes, low temp, automatic as playing). But might not work because needs to really be 120bpm.
# can also try only allowing notes that are within the currently pressed notes. (i.e. a sort of auto arpeggiator)
# can also try "fancy looper" -- i.e. start the llm with the phrase, (by default it'll loop), then affect what it does by (introducing other instruments / weighting chance of notes from diff instrument) or (weighting chance of notes of certain pitch)





























# 0
