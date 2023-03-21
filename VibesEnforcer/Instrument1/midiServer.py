import mido
# Enable "Midi 2.0" in Logic Pro X settings to stop it sending weird incorrect messages (e.g. it says its sending Sysex but sends a note_on message instead). Can also go to the Midi Input Filter in Project Settings and disable Sysex messages.
import json
import time
import asyncio
import websockets
mido.set_backend('mido.backends.rtmidi')
from collections import deque

# pc and laptop are 2s apart, so need to sync
import ntplib
import time
ntpOffset = ntplib.NTPClient().request('pool.ntp.org').offset
def ntpTime():
  return time.time() + ntpOffset

print(mido.get_input_names())
print(mido.get_output_names())

inport0 = mido.open_input("microKEY2 Air KEYBOARD")
inport1 = mido.open_input("Arturia MiniLab mkII")
# inport0 = mido.open_input("IAC Driver Bus 1")
outport0 = mido.open_output("IAC Driver Bus 2")
# Can use Logic's "External Midi" tracks to deal with forwarding instrument input and output
# Looks like either logic or Mido is dropping note_off events, it seems like logic is culprit.

msglog = deque()
msglog.clear()

msg = mido.Message('note_on', note=60, velocity=64, time=0)
outport0.send(msg)

# connect to websocket at desktop-3vakahr:8765
# ws = websocket.WebSocket()
# ws.connect("ws://desktop-3vakahr:8889")
ws = await websockets.connect("ws://desktop-3vakahr:8889")

print("Start")
selectedInstChannel = 0
while True:
  msg = None
  msg = inport0.poll()
  if msg and msg.type in ["note_on", "note_off"]:
    msg.channel = 0
  else:
    msg = inport1.poll()
    if msg and msg.type in ["note_on", "note_off"]:
      msg.channel = 1

  if msg and msg.type in ["note_on", "note_off"]:
    # Forward play input to logic on currently selected inst
    print("Processing inport note", msg)
    if msg.channel == 0:
      midoNote = mido.Message(msg.type, channel=selectedInstChannel, note=msg.note, velocity=msg.velocity)
      outport0.send(midoNote)
    # Send to server
    fullMessage = {
      "midi": {"type": msg.type, "note": msg.note, "velocity": msg.velocity, "channel": msg.channel},
      "time": ntpTime(),
    }
    await ws.send(json.dumps(fullMessage))

  # check if incoming ws message
  wsMessage = None
  try:
    wsMessage = await asyncio.wait_for(ws.recv(), timeout=0.001)
  except asyncio.TimeoutError:
    pass
  if wsMessage:
    wsMessage = json.loads(wsMessage)
    if wsMessage["type"] == "notes":
      curTime = ntpTime()
      ticksPerSecond = 500 * 2
      notes = wsMessage["notes"]
      for note in notes:
        midoNote = mido.Message(note[0], channel=note[2], note=note[3], velocity=note[4])
        dueTime = note[1]
        msglog.append({"msg": midoNote, "due": dueTime})
    elif wsMessage["type"] == "clearAllFutureForInst":
      instToClear = wsMessage["inst"]
      curTime = ntpTime()
      msglog = deque([x for x in msglog if (x["msg"].channel != instToClear or x["msg"].type == "note_off")])
    elif wsMessage["type"] == "selectInst":
      selectedInstChannel = wsMessage["inst"]
  # sort msglog by due time, since it's not guaranteed to be in order
  msglog = deque(sorted(msglog, key=lambda x: x["due"]))
  while len(msglog) > 0 and msglog[0]["due"] <= ntpTime():
    msg = msglog.popleft()["msg"]
    outport0.send(msg)
