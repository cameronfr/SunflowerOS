# netsh interface portproxy set v4tov4 listenport=8888 listenaddress=0.0.0.0 connectport=8888 connectaddress="172.21.173.58"
# netsh interface portproxy set v4tov4 listenport=8889 listenaddress=0.0.0.0 connectport=8889 connectaddress="172.21.173.58"

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

# For MIDI Server
import asyncio
import nest_asyncio
nest_asyncio.apply()
import websockets
import json
mainWebsocket = None
# Setup MIDI Server
onNoteData = lambda noteData: None
# noteData is of form e.g. {"midi": {"channel": 0, "note": 60, "velocity": 127, "type": "note_on"}, "time": 0.0}
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

# Time Syncing. PC and Laptop are 2s apart, so need to sync
import ntplib
import time
ntpOffset = None
def syncNTP():
  global ntpOffset
  ntpOffset = ntplib.NTPClient().request('pool.ntp.org').offset
def ntpTime():
  return time.time() + ntpOffset
syncNTP()

# songFMT is a list of tracks, each being a list of events.
# scoreFMT is a list of events, where each even is of the form
# ['note', deltatime, duration, channel, pitch, velocity]
# songFMT also has "patch_change" events
def songFmtToTokens(song, model2=False, num_instr_control=None, include_start_header=True):
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

  while itrack < len(song):
    for event in song[itrack]:
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
def tokensToScoreFmtNote(tokens, model2=False):
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
def tokensToScoreFmt(tokens, model2=False):
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
    note = tokensToScoreFmtNote(ss)
    note[1] += time_elapsed
    time_elapsed = note[1]
    song_f.append(note)
  return song_f
def previewScoreFmt(score, audio=True):
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
def topPFilter(logits, p):
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
# timeFmt is a list of [absolute time (ms), duration, channel, pitch, velocity]. Using this as the main format in this code.
def tokensToTimeFmt(tokens, lastNoteTimeS):
  if not tokens[-1] >= 1152:
    raise ValueError("not aligned")
  if torch.max(tokens) >= 2816:
    print("Warning: includes composition control tokens, these will be processed incorrectly")

  timeFmt = torch.zeros((len(tokens)//3, 5), dtype=torch.long)
  # Absolute time
  timeFmt[:, 0] = tokens[::3]
  timeFmt[:, 0] = torch.cumsum(timeFmt[:, 0], dim=0) * 8 # time ms
  timeFmt[:, 0] += int(lastNoteTimeS*1000) - timeFmt[-1, 0]
  # Duration
  timeFmt[:, 1] = ((tokens[1::3] -128) // 8) * 16
  # Velocity
  timeFmt[:, 4] =  (((tokens[1::3]-128) % 8)+1) * 15
  # Channel
  timeFmt[:, 2] = ((tokens[2::3] - 1152) // 128)
  # Pitch
  timeFmt[:, 3] = ((tokens[2::3] - 1152) % 128)

  startDelta = tokens[0].item() * 8

  return timeFmt, startDelta
def timeFmtToTokens(timeFmt):
  tokens = torch.zeros((timeFmt.shape[0]*3), dtype=torch.long)

  tokens[0] = 0 #startDelta // 8
  tokens[3::3] = (timeFmt[1:, 0] - timeFmt[:-1, 0]).squeeze() // 8
  # dur_vel
  tokens[1::3] = 128 + \
    8 * (torch.clip(timeFmt[:, 1] // 16, 1, 127)) + \
    torch.clip(timeFmt[:, 4], 8, 127) // 15 - 1
  # cha_pitch
  tokens[2::3] = 1152 + \
    128 * torch.clip(timeFmt[:, 2], 0, 11) + \
    torch.clip(timeFmt[:, 3], 0, 127)

  return tokens

def debugGraphTimeFmt(score, specialNoteIdxs=[], title="midi"):
  colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'pink', 'orange', 'purple', 'gray', 'green', 'gold', 'silver']
  x, y, c, m = [], [], [], []

  for i, s in enumerate(score):
    print(s[0])
    x.append((s[0] - score[0, 0]) / 1000)
    y.append(s[3])
    if i in specialNoteIdxs:
      m.append("P")
      c.append("black")
    else:
      m.append("o")
      c.append(colors[s[2]])

  plt.figure(figsize=(14,12))
  ax=plt.axes(title=title)
  ax.set_facecolor('white')
  # include piano note colored rows
  for i in range(12,96):
    plt.axhline(y=i, color='gray', alpha=0.1, linewidth=0.5)
  # color black keys slightly darker
  for i in range(12,96):
    if i % 12 in [1, 3, 6, 8, 10]:
      plt.axhline(y=i, color='gray', alpha=0.3, linewidth=4)
  # for _x, _y, _c, _m in zip(x,y,c,m):
  plt.scatter(x,y, c=c, alpha=0.8)
  # plt.scatter(x,y, c=c, marker=m)
  plt.xlabel("Time")
  plt.ylabel("Pitch")
  plt.show()

# Load Model
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

testMidi = "/home/cameronfranz/SampleTrainMidi/0a3fcc037ace7b75b8c201478f0c4656.mid"
midiBytes = open(testMidi, "rb").read()
songFmt = TMIDIX.midi2ms_score(midiBytes)
tokens = songFmtToTokens(songFmt) # song object is consumed X.X
score = tokensToScoreFmt(tokens)
previewScoreFmt(score, audio=True)

torch.set_printoptions(precision=10, sci_mode=False)
lastNoteTime = ntpTime()
timeline = tokensToTimeFmt(torch.LongTensor(tokens[6:]), lastNoteTime)[0]

def addTimeFmtNoteToTimeline(timeline, note):
  # TODO: if quantize and time matches another note, need to insert sorted by pitch
  insertLocation = torch.where(timeline[:, 0] > note[0])[0]
  if len(insertLocation) == 0:
    insertLocation = timeline.shape[0]
  else:
    insertLocation = insertLocation[0].item()
  newTimeline = torch.zeros((timeline.shape[0]+1, timeline.shape[1]), dtype=torch.long)
  newTimeline[:insertLocation] = timeline[:insertLocation]
  newTimeline[insertLocation] = torch.LongTensor(note)
  newTimeline[insertLocation+1:] = timeline[insertLocation:]
  return newTimeline



debugGraphTimeFmt(timeline)
# interesting when don't add gen notes to timeline
timeline = torch.LongTensor(0, 5)
pendingNotesBuffer = []
def onNoteData(noteData):
  pendingNotesBuffer.append(noteData)
iter = 0
temperature=0.8; top_p=0.99; promptLength =256;
while True:
  await asyncio.sleep(0.001)
  pendingPlayNotes = list(filter(lambda x: x["midi"]["channel"] == 0 and x["midi"]["type"] == "note_on", pendingNotesBuffer))
  noteWasInHighRegion = False
  if len(pendingPlayNotes) > 0:
    while len(pendingPlayNotes) > 0:
      midiEvent = pendingPlayNotes.pop(0)
      if midiEvent["midi"]["type"] == "note_on" and midiEvent["midi"]["channel"] == 0:
        if midiEvent["midi"]["note"] >= 60:
          noteWasInHighRegion = True
        duration = 2000
        selectedInstChannel = 0
        note = [int(midiEvent["time"]*1000), duration, selectedInstChannel, midiEvent["midi"]["note"], midiEvent["midi"]["velocity"]]

        timeline = addTimeFmtNoteToTimeline(timeline, note)
    pendingNotesBuffer = []

    if not noteWasInHighRegion:
      maxTimeAhead = ntpTime() + 0.26*1 # one quarter note at 120bpm
      modelInputTokens = timeFmtToTokens(timeline).unsqueeze(0).cuda()
      timelineAddition = []
      for noteNum in range(5):
        noteTokens = []
        for i in range(3):
          with torch.no_grad():
            with torch.cuda.amp.autocast(): #On: 30ms 512, 30ms 1024. Off: 30ms 512, 50ms 1024.
              if torch.max(modelInputTokens) >= 2831:
                print("Invalid token, stopping to preventing CUDA crash")
                break
              logits = model.forward(modelInputTokens[:, -1024:], return_loss=False)[:, -1, :] # remove seq dim
            # if i == 0:
              # print("The last note of modelInput is ahead of right now by", currentDueTime-ntpTime(), "biasing for notes more than", minTimeAheadMs, "ms ahead")
              # print("The last note of modelInput is ahead of the last played note by", currentDueTime-lastPlayedNoteTime)
              # minTimeAheadMs = max(0, 50 - 1000*(currentDueTime-ntpTime()))
              # logits[0][:int(minTimeAheadMs // 8)] = -1000 # at least 120ms ahead
              # TODO: this isn't relevant if model ends up predicting a note we play next
            filtered_logits = topPFilter(logits[0], top_p).unsqueeze(0) # add batch dim back so we're [batch, num_tokens]
            probs = F.softmax(filtered_logits / temperature, dim = -1)
            sampled = torch.multinomial(probs, 1)
            modelInputTokens = torch.cat((modelInputTokens, sampled), dim = -1)
            token = sampled.cpu().item()
            noteTokens.append(token)
        notes, delta = tokensToTimeFmt(torch.LongTensor(noteTokens), 0)
        note = notes[0]
        if noteNum == 0:
          note[0] = timeline[-1][0] + delta
        else:
          note[0] = timelineAddition[-1][0] + delta
        timelineAddition.append(note)

        if timelineAddition[-1][0] / 1000 > maxTimeAhead:
          print("Ahead of max time, breaking at", noteNum)
          break
      for note in timelineAddition:
        if note[2] < 12 and note[3] >= 60:
            print("Adding gen note", note)
            absTime=note[0].item()/1000; dur=note[1].item()/1000; channel=note[2].item(); pitch=note[3].item(); vel=note[4].item()
            onEvent = ["note_on", absTime, channel, pitch, vel]
            offEvent = ["note_on", absTime+dur, channel, pitch, vel]
            if note[2] not in [10,  6, 7, 9, 2, 1, 0, 11, 3, 4, 8]:
              print("Got note on unknown ch", note[3])
            else:
              msg = {"type": "notes", "notes": [onEvent, offEvent]}
              asyncio.get_event_loop().create_task(mainWebsocket.send(json.dumps(msg)))
    else:
      print("Skipped generation because note was in high region")
    print("Finished adding gen notes\n")
