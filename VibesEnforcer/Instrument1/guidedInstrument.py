# netsh interface portproxy set v4tov4 listenport=8888 listenaddress=0.0.0.0 connectport=8888 connectaddress="172.21.173.58"
#  netsh interface portproxy set v4tov4 listenport=8889 listenaddress=0.0.0.0 connectport=8889 connectaddress="172.21.173.58"

import os
import pickle
import random
import secrets
import statistics
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

# pc and laptop are 2s apart, so need to sync
import ntplib
import time
ntpOffset = ntplib.NTPClient().request('pool.ntp.org').offset
def ntpTime():
  return time.time() + ntpOffset

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

  timingTokenIdx = None
  if tokens[0] < 128:
    timingTokenIdx = 0
  elif tokens[1] < 128:
    timingTokenIdx = 1
  elif tokens[2] < 128:
    timingTokenIdx = 2
  else:
    timingTokenIdx = 3

  for i, s in enumerate(song[timingTokenIdx:]):
    # if s >= 128 and s < (12*128)+1152:
    if i % 3 != 0 and s < (12*128)+1152:
      son.append(s)
    else:
      if len(son) == 3:
        song1.append(son)
      son = []
      son.append(s)
  if len(son) == 3:
    song1.append(son)

  for ss in song1:
    note = tokensToNote(ss)
    note[1] += time_elapsed
    time_elapsed = note[1]
    song_f.append(note)
  return song_f

def previewSongFormat(score, audio=True):
  fname = "LAMC_tmp"

  detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(score, output_signature = 'Los Angeles Music Composer', output_file_name=fname, track_name='Project Los Angeles', list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0], number_of_ticks_per_quarter=500)

  x, y, c = [], [], []

  colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'pink', 'orange', 'purple', 'gray', 'green', 'gold', 'silver']

  for s in score:
    x.append(s[1] / 1000)
    y.append(s[4])
    c.append(colors[s[3]])

  if audio:
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


pendingNotesBuffer = []
currentlyPressedNotes = []
lastNoteTime = ntpTime()
def onNoteData(noteData):
  pendingNotesBuffer.append(noteData)

  pitch = noteData["midi"]["note"]
  channel = noteData["midi"]["channel"]
  if noteData["midi"]["type"] == "note_on":
    currentlyPressedNotes.append([pitch, channel])
  elif noteData["midi"]["type"] == "note_off":
    try:
      currentlyPressedNotes.remove([pitch, channel]) # we might remove them elsewhere. TODO: don't do like tis, lol
    except Exception as e:
      print("Exception when removing from notes buffer")
      pass

# if noteData["midi"]["type"] == "note_on":
#   addMidiEventToTokenInput(noteData["time"], noteData["midi"], tokenInput, lastNoteTime)
#   lastNoteTime = noteData["time"]

# MIDI Server
import asyncio
import copy
import nest_asyncio
nest_asyncio.apply()
import websockets
import json
mainWebsocket = None
# server.close()
async def handler(websocket, path):
  global mainWebsocket
  mainWebsocket = websocket
  print("testOut")
  while(True):
    data = await websocket.recv()
    noteData = json.loads(data)
    onNoteData(noteData)
server = await websockets.serve(handler, "0.0.0.0", 8889, ping_timeout=None)
task = asyncio.get_event_loop().create_task(server.serve_forever()) # doesn't print stdout or errors for some reason :/. Sometimes Need to await task to debug.
# await task

NEW_SONG_TKN = 2816

# testMidi = "/home/cameronfranz/kgSongCC0.mid"
# testMidi = "/home/cameronfranz/kgSongCC0_transposed.mid"
testMidi = "/home/cameronfranz/SampleTrainMidi/0a3fcc037ace7b75b8c201478f0c4656.mid"
# testMidi = "/home/cameronfranz/SampleTrainMidi/0a3fbd254eae9d9d7bae8985e131d493.mid"
midiBytes = open(testMidi, "rb").read()
score = TMIDIX.midi2ms_score(midiBytes)
tokens = scoreToTokens(score) # score object is consumed X.X
score2 = tokensToSongFormat(tokens)
previewSongFormat(score2)


# for i in range(5):
score = TMIDIX.midi2ms_score(midiBytes)
tokens = scoreToTokens(score, num_instr_control=9, include_start_header=True, model2=False) # score object is consumed X.X
promptLength = 256
allowStop = True
modelInput = [tokens[:promptLength]] * 1
modelInput = torch.LongTensor(modelInput).cuda()
completion = customGenerate(model, modelInput, 256, temperature=0.8, top_p=0.99, return_prime=False, min_stop_token=(NEW_SONG_TKN if allowStop else 0), verbose=True, filter_thres=0.0)
completion = completion.cpu().numpy()
songOut = tokensToSongFormat(completion[0][2:], model2=False)
previewSongFormat(songOut)

# ideas:
# "tight button" -- encourage either perfectly same onsets, or at least one quarter note
# "encourage wider chords"
# "encourage this specific chord" -- can keep track of chords / sequence being built,
controlMappingArt = {
  "instCh0": 60-1, "instCh1": 60-2, "instCh2": 60-3, "instCh3": 60-4, "instCh4": 60-5, "instCh5": 60-6, "instCh6": 60-7, "instCh7": 60-8, "instCh8": 60-9, "instCh9": 60-10, "instCh10": 60-11, "instCh11": 60-12,
  "delayZero": 60, "delaySubQtr": 61, "delayMoreQtr": 62,
  "durSubQtr": 63, "durSubWhole": 64, "durMoreWhole": 65,
  "tempLower": 66, "tempHigher": 67,
  "probsDecrease2": 68, "probsDecrease1": 69, "probsIncrease1": 70, "probsIncrease2": 71,
  "velHigher": 36, "velLower": 37, # pads
  "delayQuantize12016th": 38,
  "addInsteadOfClear": 39, # pad 4
  "addNewSongTokenInFront": 40, # pad5
  "removeBeforePrompt": 41, #pad6
  "reducePromptToLast5Seconds": 42, # pad7
  "subtleAdd": 43 # pad 8
}
controlMappingArt = {v: k for k, v in controlMappingArt.items()}

# # pregenerate biases for each note
pitchBiases = torch.zeros((128, 2831), requires_grad=False).cuda()
for i in range(1152, (12*128)+1152):
  pitch = (i-1152) % 128
  channel = (i-1152) // 128
  if channel == 9:
    # ignore drums
    continue
  pitchBiases[pitch][i] = 1

instrumentBiases = torch.zeros((128, 2831), requires_grad=False).cuda()
for i in range(1152, (12*128)+1152):
  channel = (i-1152) // 128
  instrumentBiases[channel][i] = 1


torch.set_printoptions(precision=10, sci_mode=False)
torch.tensor([1.12345e-9])


# plt.hist(currentInput[0][3::3].cpu(), bins=range(59,70))

def clearNotesByInstAndTime(tokenInput, lastNoteTime, instCh, startTime, endTime, quantize=False, allChannels=False):
  if not tokenInput[-1] >= 1152:
    raise ValueError("not aligned")
  # if quantize is True:
  #   startTime = lastNoteTime + (round((startTime-lastNoteTime)/ 0.125) * 0.125)
  #   endTime= lastNoteTime + (round((endTime-lastNoteTime)/ 0.125) * 0.125)

  # tokenInput = currentInput[0].clone()
  # lastNoteTime = currentDueTime
  # instCh = 0
  #
  # convert to easier format
  eventFormat = torch.zeros((len(tokenInput)//3, 3), dtype=torch.long)
  eventFormat[:, 0] = tokenInput[::3]
  eventFormat[:, 0] = torch.cumsum(eventFormat[:, 0], dim=0) * 8 # time ms
  startDelta = eventFormat[0:1, 0].clone()
  eventFormat[:, 1] = tokenInput[1::3] # dur_vel
  eventFormat[:, 2] = tokenInput[2::3] # cha_pitch
  eventFormat[:, 0] += int(lastNoteTime*1000) - eventFormat[-1, 0]

  sameInstMask = (eventFormat[:, 2] - 1152) // 128 == instCh
  sameInstMask |= (eventFormat[:, 2] - 1152) // 128 >= 12 # get rid of composition control tokens
  if allChannels:
    sameInstMask[:] = True
  withinTimeMask = (eventFormat[:, 0] >= int(startTime*1000)) & (eventFormat[:, 0] <= int(endTime*1000))
  # withinTimeMask = (eventFormat[:, 0] >= int((startTime-0.05)*1000)) & (eventFormat[:, 0] <= int((endTime+0.05)*1000))
  # if don't cast to int, comparisons are wrong for some reason...
  mask = sameInstMask & withinTimeMask
  eventFormat = eventFormat[~mask]

  # convert back
  tokenOutput = torch.zeros((len(eventFormat)*3), dtype=torch.long)
  # undo cumsum
  tokenOutput[0] = startDelta // 8
  tokenOutput[3::3] = (eventFormat[1:, ::3] - eventFormat[:-1, ::3]).squeeze() // 8
  # other dur_vel and cha_ptch
  tokenOutput[1::3] = eventFormat[:, 1]
  tokenOutput[2::3] = eventFormat[:, 2]

  return tokenOutput

# currentInput = torch.LongTensor([tokens[:promptLength]]).cpu()
# currentInput = currentInput[:, :255]
# currentInput[0][-1]
# previewSongFormat(tokensToSongFormat(
#   currentInput[0].cpu()
# ))

# def addMidiEventToTokenInput2(eventTime, midiEvent, tokenInput, lastNoteTime, quantize=False, duration=1.0):


# quantizes, more granular than scoreToTokens
def addMidiEventToTokenInput(eventTime, midiEvent, tokenInput, lastNoteTime, quantize=False, duration=1.0):
  # quantize to nearest 16th note, 16th note = 1/4 beat, 500ms / beat (@120bpm), 500ms / 4 = 125ms / 16th note
  # uses lastNoteTime as reference.

  quantizedTime = eventTime
  if quantize is True:
    quantizedTime = lastNoteTime + (round((eventTime-lastNoteTime)/ 0.125) * 0.125)
    # don't quantize later because won't work if note is slightly before where it should be?

  # assume tokenInput has a full token in it
  insertPos = len(tokenInput)
  cursorTime = float(lastNoteTime) #time of the note before the candidate insert position. Pytorch acting weird with the floats?
  while True:
    # print("cursorTime at pos", insertPos, "is", cursorTime)
    if insertPos == 0:
      break
    if cursorTime < quantizedTime:
      break
    if cursorTime == quantizedTime:
      pitchOfNoteBefore = (tokenInput[insertPos-1] - 1152) % 128
      if midiEvent["note"] <= pitchOfNoteBefore:
        break
    cursorTime -= float(tokenInput[insertPos-3]*8) / 1000
    # print("delta is", float(tokenInput[insertPos-3]*8) / 1000)
    insertPos -= 3

  ticksDelta = (quantizedTime - cursorTime) * 1000 # maybe quantize here instead?
  if quantize is True:
    ticksDelta = round(ticksDelta /(125))*(125)# assumes 120bpm. quantized to 16th note.

  e = [ticksDelta, duration*1000, midiEvent["channel"], midiEvent["note"], midiEvent["velocity"]]

  # Conversion
  e[0] = math.ceil(e[0] / 8)
  e[1] = math.ceil(e[1] / 16)
  # tim = max(0, min(127, e[0]))
  tim = e[0]
  dur = max(1, min(127, e[1]))
  cha = max(0, min(11, e[2]))
  ptc = max(1, min(127, e[3]))
  vel = max(8, min(127, e[4]))
  vel = round(vel / 15)
  dur_vel = (dur * 8) + (vel-1)
  cha_ptc = (cha * 128) + ptc
  tokens = [tim, dur_vel+128, cha_ptc+1152]

  # print("Adding", tokens, "cha is", cha)
  out = torch.cat([tokenInput[:insertPos], torch.LongTensor(tokens), tokenInput[insertPos:]])

  # print("desired insertion time is", eventTime, "time delta from prev is", ticksDelta/1000)
  # print("next has original delta of", out[insertPos+3]*8)
  if insertPos < len(tokenInput):
    # fix the token after. it's delta is now it's original delta minus the inserted note's delta.
    # print("fixing delta to", out[insertPos+3] - tim)
    # out[insertPos+3] = max(0, min(127, math.floor((out[insertPos+3] - tim))))
    out[insertPos+3] = max(0, min(9999999, math.floor((out[insertPos+3] - tim))))

  return out

# testTokenInput = torch.LongTensor([])
# testTokenInput = addMidiEventToTokenInput(1, {"channel": 0, "note":58, "velocity": 100}, testTokenInput, 1)
# testTokenInput
# previewSongFormat(tokensToSongFormat(testTokenInput))
# test = addMidiEventToTokenInput(0, pendingEvent["midi"], testInput[0].cpu(), 10).cuda().unsqueeze(0)
# previewSongFormat(tokensToSongFormat(test[0].cpu()))

# [0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0][8]
# if the chords aren't hitting at the same time, need to re-do the ntp time calculation on both ends.
pendingNotesBuffer = []
currentlyPressedNotes = []
temperature=0.8; top_p=0.999; promptLength =256;
# temperature=0.8; top_p=0.99; promptLength =256;
currentDueTime = ntpTime() + 1
currentNoteTokens = []
currentInput = torch.LongTensor([tokens[:promptLength]]).cuda()
# currentInput = testInput.unsqueeze(0)
# currentInput[0][0] = 249
# testInput = currentInput[0][:18*3]
selectedInstChannel = 0
previewSongFormat(tokensToSongFormat(currentInput[0][-1024:].cpu()))
for i in range(128000):
  onChaPitchToken = False
  onTimingToken = False
  onDurVelToken = False
  lastToken = currentInput[0][-1]
  if lastToken > 1152:
    onTimingToken = True
  elif (lastToken >= 128 and lastToken < 1152):
    onChaPitchToken=True
  elif lastToken < 128:
    onDurVelToken = True

  currentlyPressedKorg = [n[0] for n in currentlyPressedNotes if n[1] == 0]
  currentlyPressedArturia = [n[0] for n in currentlyPressedNotes if n[1] == 1]
  pendingPlayNotes = list(filter(lambda x: x["midi"]["channel"] == 0, pendingNotesBuffer))
  pendingControlNotes = list(filter(lambda x: x["midi"]["channel"] == 1, pendingNotesBuffer))
  modifiers = [controlMappingArt[n] for n in currentlyPressedArturia]
  statefulModifiers = [controlMappingArt[n["midi"]["note"]] for n in pendingControlNotes if n["midi"]["type"] == "note_on"]

  # if "addInsteadOfClear" not in modifiers:
  if "removeBeforePrompt" in statefulModifiers or (len(pendingPlayNotes) > 0 and "addInsteadOfClear" not in statefulModifiers):
    # mute notes while we're adding noets
    msg = {"type": "clearAllFutureForInst", "inst": selectedInstChannel}
    asyncio.get_event_loop().create_task(mainWebsocket.send(json.dumps(msg)))
  if onTimingToken and "reducePromptToLast5Seconds" in modifiers:
    print("Reducing input prompt to last 5 seconds")
    currentInput = clearNotesByInstAndTime(currentInput[0].cpu(), currentDueTime, selectedInstChannel, 0, currentDueTime-5, quantize=False, allChannels=True).cuda().unsqueeze(0)
    previewSongFormat(tokensToSongFormat(currentInput[0][-1024:].cpu()), audio=False)
    currentlyPressedNotes = []
  if onTimingToken and len(pendingPlayNotes) > 0:
    # wait until user stops pressing notes before adding them -- this way we can also get duration. For some controls keys, wait until its not being pressed anymore
    finishedAddition = len(currentlyPressedKorg) == 0 and pendingPlayNotes[-1]["midi"]["type"] == "note_off" and pendingPlayNotes[-1]["midi"]["channel"] == 0
    # if "addInsteadOfClear" not in statefulModifiers and len(pendingControlNotes) > 0:
    if len(pendingControlNotes) > 0:
      finishedAddition &= pendingControlNotes[-1]["midi"]["type"] == "note_off"
    if finishedAddition:
      relevantPending = pendingPlayNotes
      startTime = relevantPending[0]["time"]
      endTime = relevantPending[-1]["time"]

      if "addInsteadOfClear" not in statefulModifiers:
        # clear the middle area in currentInput where the notes should be insserted
        allChannels = False
        if "removeBeforePrompt" in statefulModifiers:
          allChannels = True
        currentInput = clearNotesByInstAndTime(currentInput[0].cpu(), currentDueTime, selectedInstChannel, startTime-0.05, endTime+0.05, quantize=True, allChannels=allChannels).cuda().unsqueeze(0)
        print("Clearing instead of adding")

      while len(relevantPending) > 0:
        pendingEvent = relevantPending.pop(0)
        if pendingEvent["midi"]["type"] == "note_on" and pendingEvent["midi"]["channel"] == 0:
          noteOffEvent = list(filter(lambda e: e["midi"]["type"] == "note_off" and e["midi"]["note"] == pendingEvent["midi"]["note"], relevantPending))[0]
          duration = noteOffEvent["time"] - pendingEvent["time"]
          print("Adding a note with duration", duration)
          pendingEvent["midi"]["channel"] = selectedInstChannel

          currentInput = addMidiEventToTokenInput(pendingEvent["time"], pendingEvent["midi"], currentInput[0].cpu(), currentDueTime, quantize=False, duration=duration).cuda().unsqueeze(0)

      if "removeBeforePrompt" in statefulModifiers:
        print("Clearing before prompt")
        currentInput = clearNotesByInstAndTime(currentInput[0].cpu(), currentDueTime, selectedInstChannel, 0,startTime, quantize=False, allChannels=True).cuda().unsqueeze(0)
        currentDueTime
        currentInput[0][0]=0 # change first timing token
        #Experimental
        # currentInput = torch.cat([torch.LongTensor([tokens[:258]]).cuda(), currentInput], dim=1)
      # if True or ("addInsteadOfClear" not in statefulModifiers):
      if "subtleAdd" not in statefulModifiers:
        print("Clearing extra end of model input")
        # clear to the end. Don't do before because addMidiEventToTokenInput needs an end sentinel.
        # currentInput = clearNotesByInstAndTime(currentInput[0].cpu(), currentDueTime, selectedInstChannel, endTime+0.05, currentDueTime, quantize=False).cuda().unsqueeze(0)
        currentInput = clearNotesByInstAndTime(currentInput[0].cpu(), currentDueTime, selectedInstChannel, endTime, currentDueTime, quantize=False, allChannels=True).cuda().unsqueeze(0)
        # If do endTime+0.05, timing will be more preserved, but since model is two notes ahead it will usually think the input has "ended"
        # if do this when adding notes (i.e. when addInsteadOfClear), the output becomes much less stable. Whereas, when add notes in the prompt slightly behind the frontier, it's much more stable
      if "addNewSongTokenInFront" in statefulModifiers:
        print("Adding new song token in front of prompt")
        drumsToken = 2817 # no drums
        numInstrToken = 2819 + (3 - 1) # 3 instruments
        currentInput = torch.cat([torch.LongTensor([2816, drumsToken, numInstrToken]).cuda().unsqueeze(0), currentInput], dim=1)

      #TODO: switch addMidiEventToTokenInput etc to the cleaner tensor format, so don't this
      startIdx = -3*(currentInput[0].shape[0] // 3)
      currentInput[0][startIdx::3] = torch.clip(currentInput[0][startIdx::3], 0, 127) # clip because addMidiEventToTokenInput uses >2s timings as intermediate step

      pendingNotesBuffer = []
      previewSongFormat(tokensToSongFormat(currentInput[0][-1024:].cpu()), audio=False)

  with torch.no_grad():
    if torch.max(currentInput) >= 2831:
      print("Invalid token, stopping to preventing CUDA crash")
      break
    with torch.cuda.amp.autocast(): #On: 30ms 512, 30ms 1024. Off: 30ms 512, 50ms 1024.
    # logits = model.forward(currentInput[:, -model.max_seq_len:], return_loss=False)[:, -1, :] # remove seq dim
      logits = model.forward(currentInput[:, -1024:], return_loss=False)[:, -1, :] # remove seq dim


  NO_NOTE_REPEAT_KEY = 42# pad 7
  CHANGE_SONG_KEY = -1 #43 #pad 8

  modified_logits = logits.clone()
  biasedTokensTemp = temperature
  biasMask = torch.ones(2831, dtype=torch.bool).cuda()
  logitAddition = 6
  if "probsDecrease2" in modifiers:
    logitAddition = -10
  if "probsDecrease1" in modifiers:
    logitAddition = -2
  if "probsIncrease1" in modifiers:
    logitAddition = 10
  if "probsIncrease2" in modifiers:
    logitAddition = 12
  if "tempLower" in modifiers:
    biasedTokensTemp = 0.2
  if "tempHigher" in modifiers:
    biasedTokensTemp = 0.9
  if onChaPitchToken:
    # ignore pitch biases for now
    # pitchBiasesMask = torch.sum(pitchBiases[currentlyPressedKorg], axis=0) != 0
    # if torch.sum(pitchBiasesMask) > 0:
    #   biasMask = biasMask & pitchBiasesMask

    # Instruments bias. If none pressed, do nothing. If multiple pressed, bias towards union of those instruments.
    instBiasesMask = torch.zeros(2831, dtype=torch.bool).cuda()
    for i in range(12):
      if "instCh"+str(i) in modifiers:
        print("Set selected instrument to", i)
        selectedInstChannel = i
        msg = {"type": "selectInst", "inst": selectedInstChannel}
        asyncio.get_event_loop().create_task(mainWebsocket.send(json.dumps(msg)))
        instBiasesMask = instBiasesMask | (instrumentBiases[i] != 0)
    if torch.sum(instBiasesMask) > 0:
      biasMask = biasMask & instBiasesMask

    if NO_NOTE_REPEAT_KEY in currentlyPressedArturia:
      lastChaPitchToken = currentInput[0][-3]
      lastPitch = (lastChaPitchToken-1152) % 128
      print("Biasing away from repeat, last pitch was", lastPitch, str(ntpTime()), end="\r")
      modified_logits[0][torch.where(pitchBiases[lastPitch] != 0)] = -2000
  if onTimingToken:
    modified_logits[0][NEW_SONG_TKN] -= 1000
    if CHANGE_SONG_KEY in currentlyPressedArturia:
      print("Trying to change song" + str(ntpTime()), end="\r")
      modified_logits[0][NEW_SONG_TKN] += 1010

    # Union pressed desired times, then bias towards those times.
    # But maybe it should be more like, "delayZeroUp", "delayZeroDown" ... because right now, cant say "increase prob of delayZero, but decrease prob of delaySubQuarter"
    timingBiasMask = torch.zeros(2831, dtype=torch.bool).cuda()
    if "delayZero" in modifiers:
      timingBiasMask[:4] = True
    if "delaySubQuarter" in modifiers:
      timingBiasMask[4:66] = True # one qtr note at 120bpm is 500ms. 500ms / 8ms = 62
    if "delayMoreQtr" in modifiers:
      timingBiasMask[66:128] = True
    if "delayQuantize12016th" in modifiers:
      # timingBiasMask[0:2] = True
      # timingBiasMask[8] = True #32nd
      timingBiasMask[15:17] = True # 16th
      timingBiasMask[30:33] = True # 8th
      timingBiasMask[45:48] = True # 8th+16th (8dot)
      timingBiasMask[62:64] = True # quarter
      timingBiasMask[76:80] = True # quarter + 16th
      timingBiasMask[91:95] = True # quarter + 8th
      timingBiasMask[108:111] = True
      timingBiasMask[123:127] = True
    if torch.sum(timingBiasMask) > 0:
      biasMask = biasMask & timingBiasMask

  if onDurVelToken:
    # 128 duration x 8 vel. Dur is x16 in ms
    durBiasMask = torch.zeros(2831, dtype=torch.bool).cuda()
    if "durSubQtr" in modifiers:
      durBiasMask[128:128+34*8] = True # 34*16 = 544ms
    if "durSubWhole" in modifiers:
      durBiasMask[128+34*8:128+102*8] = True # 102*16 = 1632ms (3 qtr notes ish)
    if "durMoreWhole" in modifiers:
      durBiasMask[128+102*8:128+1024] = True # model only has up to 128*16 = 2048ms, doesn't have longer durations. So this is 3qtr notes+ durations.
    if torch.sum(durBiasMask) > 0:
      biasMask = biasMask & durBiasMask

    velBiasMask = torch.zeros(2831, dtype=torch.bool).cuda()
    if "velLower" in modifiers:
      velBiasMask[128:128+1024][0::8] = True # vel 15
      velBiasMask[128:128+1024][1::8] = True # vel 30
    if "velHigher" in modifiers:
      velBiasMask[128:128+1024][6::8] = True # vel 90
      velBiasMask[128:128+1024][7::8] = True # vel 105
    if torch.sum(velBiasMask) > 0:
      biasMask = biasMask & velBiasMask

  print("modifiers are", modifiers, "notes are", currentlyPressedKorg, "biasMask size is", torch.sum(biasMask), "logitAddition is", logitAddition, "biasedTokensTemp is", biasedTokensTemp, end="\r")
  if (~biasMask).sum() != 0:
    modified_logits[0][biasMask] += logitAddition
  temperatureArr = torch.ones(2831, dtype=torch.float).cuda() * temperature
  temperatureArr[biasMask] = biasedTokensTemp
  filtered_logits = top_p_filter(modified_logits[0], top_p).unsqueeze(0) # add batch dim back so we're [batch, num_tokens]
  probs = F.softmax(filtered_logits / temperatureArr.unsqueeze(0), dim = -1)
  sampled = torch.multinomial(probs, 1)
  currentInput = torch.cat((currentInput, sampled), dim = -1)

  token = sampled.cpu().item()
  if onChaPitchToken:
    tokenCh = (token-1152) // 128
    tokenPitch = (token-1152) % 128
    for i in range(12):
      if "instCh"+str(i) in modifiers and "delayZero" in modifiers:
        if tokenCh == i:
          print("Introduced desired instrument, clearing")
          currentlyPressedNotes = []
    # if "delayZero" in modifiers and len(currentlyPressedKorg) > 0:
    #   if len(currentNoteTokens) > 0:
    #     timingToken = currentNoteTokens[0]
    #     if timingToken < 4 and tokenPitch in currentlyPressedKorg:
    #       print("Introduced desired zero-delay note, clearing that note")
    #       if len(currentlyPressedKorg) == 1:
    #         currentlyPressedNotes = []
    #       else:
    #         currentlyPressedNotes.remove([tokenPitch, 0])
    # if len(currentNoteTokens) > 0:
      # if tokenPitch in currentlyPressedKorg:
      #   print("Introduced desired note, clearing that note")
      #   if len(currentlyPressedKorg) == 1:
      #     currentlyPressedNotes = []
      #   else:
      #     currentlyPressedNotes.remove([tokenPitch, 0])




  # if len(currentlyPressedKorg) > 0:
  #   if onChaPitchToken:
  #     # print("Probabilities of biased notes are now", torch.max(probs[0][tokensToBiasIndices].cpu(), dim=0)[0], end="\r")
  #     pitch = (token-1152) % 128
  #     # print("Sampled note with pitch", pitch)
  #     if token in tokensToBiasIndices[0].tolist():
  #       print("Sampled a biased note!")
  #       # Remove until it is pressed again
  #       for p in range(pitch % 12, 128, 12):
  #         if [p, 0] in currentlyPressedNotes:
  #           print("removing", pitch, p)
  #           currentlyPressedNotes.remove([p, 0])
  #     if pitch in currentlyPressedKorg:
  #       # print("Sampled note with pitch", pitch, "removing")
  #       # currentlyPressedNotes.remove([pitch, 0])
  #       pass
  # if "delayZero" in modifiers:
  #   if onTimingToken:
  #     if token < 4:
  #       print("Got delay 0, clearing")
  #       currentlyPressedNotes = []
  # if ENCOURAGE_INSTR_KEY in currentlyPressedArturia:
  #   if onChaPitchToken:
  #     if (token-1152) // 128 == 11:
  #       print("Encouraged instrument")
  #       currentlyPressedNotes.remove([ENCOURAGE_INSTR_KEY, 1])

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
        if note[3] not in [10,  6, 7, 9, 2, 1, 0, 11, 3, 4, 8]:
          print("Got note on ch", note[3])
        else:
          msg = {"type": "notes", "notes": [onEvent, offEvent]}
          asyncio.get_event_loop().create_task(mainWebsocket.send(json.dumps(msg)))
  # note =['note', deltatime, duration, channel, pitch, velocity]
  timeNow = ntpTime()
  if currentDueTime <timeNow + 0.01:
    print("Model couldn't keep up", end="\n")
  if currentDueTime < timeNow + 0.1:
    print("note has less than 0.1 seconds buffer, adding time")
    currentDueTime += 0.3 # todo: model isn't aware of these changes
  if currentDueTime > timeNow + 0.8:
    # print("note has more than 0.8 seconds buffer", currentDueTime - ntpTime())
    await asyncio.sleep(0.2)
    # currentDueTime = ntpTime() + 2 # todo: model isn't aware of these changes
  if currentDueTime > timeNow + 1.5:
    print("More than 1.5 seconds in buffer")
    await asyncio.sleep(0.3)
  await asyncio.sleep(0.0)


pendingNotesBuffer = []
currentlyPressedNotes = []
currentInput = torch.LongTensor([[]]).cuda()
currentDueTime = ntpTime()
previewSongFormat(tokensToSongFormat(currentInput[0][-1024:].cpu()), audio=True)
iter = 0
while True:
  await asyncio.sleep(0.001)
  pendingPlayNotes = list(filter(lambda x: x["midi"]["channel"] == 0 and x["midi"]["type"] == "note_on", pendingNotesBuffer))
  noteWasInHighRegion = False
  if len(pendingPlayNotes) > 0:
    while len(pendingPlayNotes) > 0:
      pendingEvent = pendingPlayNotes.pop(0)
      if pendingEvent["midi"]["type"] == "note_on" and pendingEvent["midi"]["channel"] == 0:
        if pendingEvent["midi"]["note"] >= 60:
          noteWasInHighRegion = True
        print("Added played note to history")
        # noteOffEvent = list(filter(lambda e: e["midi"]["type"] == "note_off" and e["midi"]["note"] == pendingEvent["midi"]["note"], pendingPlayNotes))[0]
        # duration = noteOffEvent["time"] - pendingEvent["time"]
        duration = 0.25
        pendingEvent["midi"]["channel"] = selectedInstChannel
        currentInput = addMidiEventToTokenInput(pendingEvent["time"], pendingEvent["midi"], currentInput[0].cpu(), currentDueTime, quantize=False, duration=duration).cuda().unsqueeze(0)
        currentDueTime = pendingEvent["time"]
    pendingNotesBuffer = []

    # if currentInput.shape[1] == 0:
    #   continue
    if noteWasInHighRegion:
      print("note was in high region, skipping")
      continue

    startIdx = -3*(currentInput[0].shape[0] // 3)
    currentInput[0][startIdx::3] = torch.clip(currentInput[0][startIdx::3], 0, 127) # clip because addMidiEventToTokenInput uses >2s timings as intermediate step

    modelInputTmp = currentInput.clone()
    currentDueTimeTmp = currentDueTime #ntpTime()+0.500 #currentDueTime
    maxTimeAhead = currentDueTimeTmp + 0.26
    for noteNum in range(2):
      noteTokens = []
      for i in range(3):
        with torch.no_grad():
          if torch.max(currentInput) >= 2831:
            print("Invalid token, stopping to preventing CUDA crash")
            break
          with torch.cuda.amp.autocast(): #On: 30ms 512, 30ms 1024. Off: 30ms 512, 50ms 1024.
            logits = model.forward(modelInputTmp[:, -1024:], return_loss=False)[:, -1, :] # remove seq dim
          if noteNum == 0 and i == 0:
            logits[0][:15] = -1000 # at least 120ms ahead
          filtered_logits = top_p_filter(logits[0], top_p).unsqueeze(0) # add batch dim back so we're [batch, num_tokens]
          probs = F.softmax(filtered_logits / temperatureArr.unsqueeze(0), dim = -1)
          sampled = torch.multinomial(probs, 1)
          # if noteNum == 0 and i == 0:
          #   sampled = torch.LongTensor([[63]]).cuda()
          modelInputTmp = torch.cat((modelInputTmp, sampled), dim = -1)
          token = sampled.cpu().item()
          noteTokens.append(token)
      note = tokensToNote(noteTokens, model2=False)
      if note[3] < 12 and note[4] >= 60:
        deltaTime = note[1] / 1000
        duration = note[2] / 1000
        currentDueTimeTmp += deltaTime
        currentDueTime += deltaTime
        # note =['note', deltatime, duration, channel, pitch, velocity]
        # if note[4] >= 60:
        print("Deltatime is", deltaTime, "duration is", duration, "pitch is", note[4], "velocity is", note[5])
        currentInput = torch.cat((currentInput, torch.LongTensor([noteTokens]).cuda()), dim = -1)
        onEvent = ["note_on", currentDueTimeTmp, note[3], note[4], note[5]]
        offEvent = ["note_off", currentDueTimeTmp+duration, note[3], note[4], note[5]]
        if note[3] not in [10,  6, 7, 9, 2, 1, 0, 11, 3, 4, 8]:
          print("Got note on ch", note[3])
        else:
          msg = {"type": "notes", "notes": [onEvent, offEvent]}
          asyncio.get_event_loop().create_task(mainWebsocket.send(json.dumps(msg)))
      if currentDueTimeTmp > maxTimeAhead:
        print("ahead of max time, breaking at", n)
        break
    # if iter % 10 == 0:
      # previewSongFormat(tokensToSongFormat(currentInput[0][-1024:].cpu()), audio=False)





# can try realtime note accompaniment (e.g. only 2 notes, low temp, automatic as playing). But might not work because needs to really be 120bpm.
# can also try only allowing notes that are within the currently pressed notes. (i.e. a sort of auto arpeggiator)
# can also try "fancy looper" -- i.e. start the llm with the phrase, (by default it'll loop), then affect what it does by (introducing other instruments / weighting chance of notes from diff instrument) or (weighting chance of notes of certain pitch)





























# 0
