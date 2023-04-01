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
%matplotlib inline

# %cd ../
# %cd "composer assistant"
# import unjoined_vocab_tokenizer as ujt
# import transformers

# Composers Assistant stuff
# model_path = os.path.join("models/unjoined/infill", 'finetuned_epoch_18_0', 'model')
# caUnjoinedTokenizer = ujt.UnjoinedTokenizer('unjoined_include_note_duration_commands')
# caModel = transformers.T5ForConditionalGeneration.from_pretrained(model_path).cuda()
# caModel.eval()
#
# def timeFmtToTokensCA(timeFmt):
#   timeFmt = timeline.clone()
#   timeFmt[1:, 0] = timeFmt[1:, 0] - timeFmt[:-1, 0]
#   timeFmt[0, 0] = 0
#
#   tokenStrings = [";I:0"]
#   curTicks = 0
#   tokenStrings
#   tokenStrings
#   %%timeit
#   for i in range(timeFmt.shape[0]):
#     note = timeFmt[i]
#     waitTime = int(note[0].item() // (2.6404*8))
#     duration = int(note[1].item() // (2.6404*8))
#     pitch = note[3]
#     tokenStrings.append(";w:{}".format(waitTime))
#     tokenStrings.append(";d:{}".format(duration))
#     tokenStrings.append(";N:{}".format(pitch))
#   tokens = caUnjoinedTokenizer.encode("".join(tokenStrings))
#   input_ids = torch.stack([torch.tensor(tokens, dtype=torch.long)]).cuda()
#
#   %%timeit
#   out = caModel.generate(input_ids=input_ids,
#                           num_return_sequences=1,
#                           do_sample=True,
#                           temperature=0.7,
#                           # remove_invalid_values=True,
#                           # top_k=100,
#                           top_p=0.99,
#                           min_length=1,
#                           max_new_tokens=3,
#                           decoder_start_token_id=caUnjoinedTokenizer.pad_id(),
#                           pad_token_id=caUnjoinedTokenizer.pad_id(),
#                           bos_token_id=caUnjoinedTokenizer.bos_id(),
#                           eos_token_id=caUnjoinedTokenizer.eos_id(),
#                           use_cache=True,
#                           # force_words_ids=forced_ids,
#                           # encoder_no_repeat_ngram_size=enc_no_repeat_ngram_size,
#                           # repetition_penalty=1.01
#                           )
#     out
#
#   # gonna be a PITA, maybe try in reaper first.

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
server = None
async def restartMidiServer():
  global server
  if server is not None:
    server.close()
  server = await websockets.serve(handler, "0.0.0.0", 8889, ping_timeout=None)
  task = asyncio.get_event_loop().create_task(server.serve_forever()) # doesn't print stdout or errors for some reason :/. Sometimes Need to await task to debug.
await restartMidiServer()

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
  tokens[3::3] = torch.clip(tokens[3::3], 0, 127)
  # dur_vel
  tokens[1::3] = 128 + \
    8 * (torch.clip(timeFmt[:, 1] // 16, 1, 127)) + \
    torch.clip(timeFmt[:, 4], 8, 127) // 15 - 1
  # cha_pitch
  tokens[2::3] = 1152 + \
    128 * torch.clip(timeFmt[:, 2], 0, 11) + \
    torch.clip(timeFmt[:, 3], 0, 127)

  return tokens

def debugGraphTimeFmt(scores, title="midi", dotColors=None, showGraph=True, dotShapes=None):
  colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'pink', 'orange', 'purple', 'gray', 'green', 'gold', 'silver']

  plt.figure(figsize=(7,6))
  ax=plt.axes(title=title)
  ax.set_facecolor('white')
  # include piano note colored rows
  for i in range(12,96):
    plt.axhline(y=i, color='gray', alpha=0.1, linewidth=0.5)
  # color black keys slightly darker
  for i in range(12,96):
    if i % 12 in [1, 3, 6, 8, 10]:
      plt.axhline(y=i, color='gray', alpha=0.3, linewidth=4)

  startTime = min([s[0, 0].item() for s in scores if len(s) > 0])

  for idx, score in enumerate(scores):
    x, y, c, m = [], [], [], []
    for i, s in enumerate(score):
      x.append((s[0] - startTime) / 1000)
      y.append(s[3])
      m.append("o")
      if dotColors is None:
        c.append(colors[s[2]])
      else:
        c.append(dotColors[idx])
    plt.scatter(x,y, c=c, marker=dotShapes[idx], s=160, alpha=0.7)

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

def addTimeFmtNoteToTimeline(timeline, note, quantizeIfSimulataneousHit=False):
  # TODO: channel order when same pitch, same time note comes in is not consistent.

  # timeline = testAdded.clone()
  # debugGraphTimeFmt(timeline)
  # note = [1679963283666, 500, 0, 50, 51]; quantizeIfSimulataneousHit=True

  insertLocation = torch.where(timeline[:, 0] > note[0])[0]
  if len(insertLocation) == 0:
    insertLocation = timeline.shape[0]
  else:
    insertLocation = insertLocation[0].item()

  duplicateTimeThresh = 150
  nearbyNotes = timeline[torch.abs(timeline[:, 0]-note[0]) < duplicateTimeThresh]
  duplicatesMask = (nearbyNotes[:, 2] == note[2]) & (nearbyNotes[:, 3] == note[3])
  if torch.sum(duplicatesMask) > 0:
    print("Duplicate note (or, within 150), not inserting") # this thresh will also prevent 1/16th note spam of same note, unforunately
    return timeline, -1

  simultTimeThresh = 62.5
  if quantizeIfSimulataneousHit:
    if insertLocation > 0 and (abs(timeline[insertLocation-1, 0] - note[0]) < simultTimeThresh):
      note[0] = timeline[insertLocation-1, 0]
      while (insertLocation > 0) and (timeline[insertLocation-1, 0] == note[0]) and (timeline[insertLocation-1, 3] < note[3]):
        insertLocation -= 1
    elif insertLocation < timeline.shape[0] and (abs(timeline[insertLocation, 0] - note[0]) < simultTimeThresh):
      note[0] = timeline[insertLocation, 0]
      while (insertLocation < timeline.shape[0]) and (timeline[insertLocation, 0] == note[0]) and (timeline[insertLocation, 3] > note[3]):
        insertLocation += 1
    # recalculate insertLocation. pitch is descending. chan is same order, but not set?

  newTimeline = torch.zeros((timeline.shape[0]+1, timeline.shape[1]), dtype=torch.long)
  newTimeline[:insertLocation] = timeline[:insertLocation]
  newTimeline[insertLocation] = torch.LongTensor(note)
  newTimeline[insertLocation+1:] = timeline[insertLocation:]
  return newTimeline, insertLocation


debugGraphTimeFmt(timeline[-60:])
previewScoreFmt(tokensToScoreFmt(timeFmtToTokens(timeline[:])), audio=True)
testAdded = timeline.clone()
testAdded = torch.LongTensor(0, 5)
testAdded, _ = addTimeFmtNoteToTimeline(testAdded, [1679963283384, 500, 0, 46, 49], quantizeIfSimulataneousHit=True)
addTimeFmtNoteToTimeline(testAdded, [1679963283666, 500, 0, 50, 51], quantizeIfSimulataneousHit=True)
testAdded
debugGraphTimeFmt(timeline)
timeline

# TODO: model should be able to get this chord+descending thing down.
# TODO: why double notes? / the slighly jank timings on the responses.
# TODO: quantize 120bpm?
# TODO: why is it hard to introduce / use other instruments effectively? muting/response region wrong?
# TODO: prob need to mute both regions when have both hands on stuff. Maybe keep muting that region when the notes are still being held.
#TODO: I think double notes cause model to spiral

pendingNotesBuffer = []
controlNotesPressed = []
selectedInstChannel = 0
def onNoteData(noteData):
  global selectedInstChannel
  pendingNotesBuffer.append(noteData)
  if noteData["midi"]["type"] == "note_on" and noteData["midi"]["channel"] == 1:
    pitch = noteData["midi"]["note"]
    controlNotesPressed.append(pitch)
    if pitch in range(48, 60):
      selectedInstChannel = pitch - 48
      msg = {"type": "selectInst", "inst": selectedInstChannel}
      asyncio.get_event_loop().create_task(mainWebsocket.send(json.dumps(msg)))
  if noteData["midi"]["type"] == "note_off" and noteData["midi"]["channel"] == 1:
    pitch = noteData["midi"]["note"]
    if pitch in controlNotesPressed:
      controlNotesPressed.remove(pitch)
# await restartMidiServer()

syncNTP()
timeline = torch.LongTensor(0, 5)
iter = 0
# pad1-36: disable reponse gen everywhere | pad2-37: generate without input | pad3-38 -- response region lows | pad4-39: reponse region everywhere | pad5-40 -- ignore input | pad6-41 -- insert at frontier | pad7-42  | pad8-43 -- clear current input
# temperature=0.8; top_p=0.999;
temperature=0.7; top_p=0.99;
# temperature=0.1; top_p=0.99;
userRecentlyPlayedNotesList = []
responseRegion = list(range(60, 128)) + list(range(0, 36))
unfinishedNotes = {} # pitch : idx -- ignornig chan for now
pendingNotesBuffer = []
debugGraphs = []
responseRegionPredNotes = torch.LongTensor(0, 5)
# responseRegionPredNotes on works well with max 0.52 second pred-ahead (i.e. pad 6)
while True:
  await asyncio.sleep(0.001)
  pendingPlayNotes = list(filter(lambda x: x["midi"]["channel"] == 0 and (x["midi"]["type"] == "note_on" or x["midi"]["type"] == "note_off"), pendingNotesBuffer))
  userPlayedNoteInResponseRegion = False
  if (43 in controlNotesPressed and (len(unfinishedNotes.keys()) == 0 and timeline.shape[0] != 0)):
    print("Resetting timeline / prompt")
    timeline = torch.LongTensor(0, 5)
    unfinishedNotes = {}
  hasEnoughNotesWhenStarting = (timeline.shape[0] == 0 and (len(pendingPlayNotes) > 3)) or (timeline.shape[0] != 0)
  # if (len(pendingPlayNotes) > 0 or (37 in controlNotesPressed) or (len(unfinishedNotes.keys()) > 0)) and hasEnoughNotesWhenStarting:
  if (len(pendingPlayNotes) > 0 or (37 in controlNotesPressed)) and hasEnoughNotesWhenStarting:
    debugGraphs.append({})
    if len(pendingPlayNotes) > 0:
      debugGraphs[-1]["timelineBeforePlayedNotes"] = timeline.clone()
      responseRegionPredNotes = responseRegionPredNotes[responseRegionPredNotes[:, 0] > int(pendingPlayNotes[-1]["time"]*1000)]
      if (41 in controlNotesPressed):
        # aheadNotes = timeline[:, 0] > (int(pendingPlayNotes[0]["time"] * 1000)) + 0 #0.26*2# clear notes more than 0ms ahead
        aheadNotes = timeline[:, 0] > int(pendingPlayNotes[0]["time"] * 1000) + 0.26*2# clear notes more than 1 quarter note ahead
        timeline = timeline[~aheadNotes]
        print("Cleared", torch.sum(aheadNotes), "ahead notes")
        if (torch.sum(aheadNotes) > 0):
          msg = {"type": "clearAllFutureForInst", "inst": -1}
          asyncio.get_event_loop().create_task(mainWebsocket.send(json.dumps(msg)))
        # Clearing ahead is good when want played notes to have big effect. But removes chance of playing "with" the model
        # Takes out one or two notes max (because generation stops when there are pending play notes)
      while len(pendingPlayNotes) > 0 and not (40 in controlNotesPressed):
        midiEvent = pendingPlayNotes.pop(0)
        if midiEvent["midi"]["type"] == "note_off":
          pitch = midiEvent["midi"]["note"]
          if pitch in unfinishedNotes:
            noteIdx = unfinishedNotes[pitch]
            note = timeline[noteIdx]
            duration = int(midiEvent["time"] * 1000) - note[0]
            timeline[noteIdx][1] = duration
            print("Updated duration to", duration)
            del unfinishedNotes[midiEvent["midi"]["note"]]
        if midiEvent["midi"]["type"] == "note_on":
          if midiEvent["midi"]["note"] in responseRegion and (not 42 in controlNotesPressed):
            userPlayedNoteInResponseRegion = True
          duration = torch.mean(timeline[-16:, 1].to(torch.float))
          if timeline.shape[0] == 0:
            duration = 500
          note = [int(midiEvent["time"]*1000), int(duration), selectedInstChannel, midiEvent["midi"]["note"], midiEvent["midi"]["velocity"]]
          print("Adding played note", note)
          timeline, insertLocation = addTimeFmtNoteToTimeline(timeline, note, quantizeIfSimulataneousHit=True)
          # I feel like quantizing will put is into realm of non-humanized midis / non-performance capture midis
          if insertLocation != -1:
            unfinishedNotes[note[3]] = insertLocation

          # Strat 1: +- 12 notes on all recently played keys.
          # userRecentlyPlayedNotesList.append(note)
          # userRecentlyPlayedNotes = torch.LongTensor(userRecentlyPlayedNotesList[-30:])
          # pitches = userRecentlyPlayedNotes[abs(userRecentlyPlayedNotes[:, 0]-int(ntpTime()*1000)) < 2100][:, 3]
          # responseRegionInv = set()
          # for p in pitches:
          #   responseRegionInv = set(range(p -12, p+12)).union(responseRegionInv)
          # Strat 2: 2-octave block of last played notes
          # pitch = note[3]
          # responseRegionInv = set(range(12 + 24 * ((pitch - 12)// 24), 36 + 24 * ((pitch - 12)// 24)))
          # responseRegion = list(set(range(0,128)) - set(responseRegionInv))

          # Strat 3: last played notes and any held notes, in 2 octav eblocks
          responseRegionInv = set()
          for pitch in unfinishedNotes.keys():
            responseRegionInv = set(range(12 + 24 * ((pitch - 12)// 24), 36 + 24 * ((pitch - 12)// 24))).union(responseRegionInv)
            # responseRegionInv = set(range(0 + 12* ((pitch)// 12), 12 + 12 * ((pitch)// 12))).union(responseRegionInv)
          responseRegion = list(set(range(0,128)) - set(responseRegionInv))
          print("Blocked-from-response area size is", len(responseRegionInv))

      pendingNotesBuffer = []
    # responseRegion = list(range(60, 128)) + list(range(0, 36))
    if 38 in controlNotesPressed:
      responseRegion = list(range(0, 60))
    elif 39 in controlNotesPressed:
      responseRegion = list(range(0, 128))

    debugGraphs[-1]["timelineAfterPlayedNotes"] = timeline.clone()

    if (not userPlayedNoteInResponseRegion) and (not 36 in controlNotesPressed):
      maxTimeAhead = ntpTime() + 0.26*4 # one quarter note at 120bpm
      maxTimeBehind = 1
      modelInputTimeline = timeline.clone()
      # for n in responseRegionPredNotes:
      #   modelInputTimeline, _ = addTimeFmtNoteToTimeline(modelInputTimeline, n, quantizeIfSimulataneousHit=True)
      debugGraphs[-1]["responseRegionPredNotes"] = responseRegionPredNotes.clone()
      debugGraphs[-1]["modelInputTimeline"] = modelInputTimeline.clone()
      modelInputTokens = timeFmtToTokens(modelInputTimeline).unsqueeze(0).cuda()
      timelineAddition = []
      for noteNum in range(16):
        await asyncio.sleep(0.001)

        frontierNote = timeline[-1] if noteNum == 0 else timelineAddition[-1]
        # be careful with dividing longs in pytorch -- unintuitive
        if (frontierNote[0].item() / 1000) > maxTimeAhead:
          print("Ahead of max time, breaking at", noteNum)
          break
        if not (37 in controlNotesPressed) and ((frontierNote[0].item() / 1000) < (ntpTime() - maxTimeBehind)):
          print("Too far behind, breaking")
          break
        pendingPlayNotes = list(filter(lambda x: x["midi"]["channel"] == 0 and x["midi"]["type"] == "note_on", pendingNotesBuffer))
        if len(pendingPlayNotes) > 0:
          print("Have pending notes, breaking at", noteNum)
          break
        noteTokens = []
        for i in range(3):
          with torch.no_grad():
            with torch.cuda.amp.autocast(): #On: 30ms 512, 30ms 1024. Off: 30ms 512, 50ms 1024.
              if torch.max(modelInputTokens) >= 2831:
                print("Invalid token, stopping to preventing CUDA crash")
                break
              logits = model.forward(modelInputTokens[:, -1024:], return_loss=False)[:, -1, :] # remove seq dim
            # if i == 0:
              # Strat 1: only predict notes far enough ahead to be played
              # currentDueTime = frontierNote[0].item() / 1000
              # timeAvailableMs = 1000*(currentDueTime-ntpTime())
              # minTimeAheadMs = min(500, max(0, 100 - timeAvailableMs)) # it takes approx 33*3 for one note
              # print("The last note of modelInput is ahead of right now by", timeAvailableMs, "biasing for notes more than", minTimeAheadMs, "ms ahead")
              # if minTimeAheadMs > 0:
              #   logits[0][:int(minTimeAheadMs // 8)] = -1000
              # TODO: this isn't relevant if model ends up predicting a note we play next
            logits[0, 2816] = -1000 # new song token
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
      for note in timelineAddition:
        withinTime = note[0].item()/1000.0 > ntpTime() + 0.02
        # withinTime = True
        if note[2] < 12 and (note[3] in responseRegion or note[2] != selectedInstChannel) and withinTime:
          print("Adding gen note to timeline, and playing", note.tolist())
          timeline, _ = addTimeFmtNoteToTimeline(timeline, note, quantizeIfSimulataneousHit=True)
          absTime=note[0].item()/1000; dur=note[1].item()/1000; channel=note[2].item(); pitch=note[3].item(); vel=note[4].item()
          onEvent = ["note_on", absTime, channel, pitch, vel]
          offEvent = ["note_off", absTime+dur, channel, pitch, vel]
          if note[2] not in [10,  6, 7, 9, 2, 1, 0, 11, 3, 4, 8, 5]:
            print("Got note on unknown ch", note[3])
          else:
            msg = {"type": "notes", "notes": [onEvent, offEvent]}
            asyncio.get_event_loop().create_task(mainWebsocket.send(json.dumps(msg)))
        else:
          if not withinTime:
            print("Note not added or played because it would play too late")
          elif note[3] not in responseRegion:
            responseRegionPredNotes, _ = addTimeFmtNoteToTimeline(responseRegionPredNotes, note, quantizeIfSimulataneousHit=True)
      if len(timelineAddition) > 0:
        debugGraphs[-1]["timelineAddition"] = torch.stack(timelineAddition).clone()
      debugGraphs[-1]["timelineAfterGenNotes"] = timeline.clone()
    else:
      print("Skipped generation because note was in high region or controlNote")
    print("Finished adding gen notes\n")


for d in debugGraphs[-30:]:
  if "timelineAddition" not in d:
    d["timelineAddition"] = []
  if "timelineAfterGenNotes" not in d:
    d["timelineAfterGenNotes"] = []
  l = 30
  debugGraphTimeFmt([d["timelineBeforePlayedNotes"][-l:], d["timelineAfterPlayedNotes"][-l:], d["timelineAfterGenNotes"][-l:], d["timelineAddition"][-l:], d["responseRegionPredNotes"][-l:]], dotColors=["black", "red", "green", "blue", "orange"], dotShapes=["1", "2", "3", "4", "o"])
