import mido
# Enable "Midi 2.0" in Logic Pro X settings to stop it sending weird incorrect messages (e.g. it says its sending Sysex but sends a note_on message instead). Can also go to the Midi Input Filter in Project Settings and disable Sysex messages.
import json
import time
import asyncio
import websockets
mido.set_backend('mido.backends.rtmidi')
from collections import deque
!pip show websocket-client

print(mido.get_input_names())
print(mido.get_output_names())

# inport0 = mido.open_input("LUMI Keys BLOCK")
# inport1 = mido.open_input("Arturia MiniLab mkII")
inport0 = mido.open_input("IAC Driver Bus 1")
outport0 = mido.open_output("IAC Driver Bus 2")
# Can use Logic's "External Midi" tracks to deal with forwarding instrument input and output

msglog = deque()
msglog.clear()

msg = mido.Message('note_on', note=60, velocity=64, time=0)
outport0.send(msg)

# connect to websocket at desktop-3vakahr:8765
# ws = websocket.WebSocket()
# ws.connect("ws://desktop-3vakahr:8889")
ws = await websockets.connect("ws://desktop-3vakahr:8889")


print("Start")
#Rewritten for new version of mido
while True:
  msg = None
  msg = inport0.poll()

  if msg and msg.type in ["note_on", "note_off"]:
    fullMessage = {
      "midi": {"type": msg.type, "note": msg.note, "velocity": msg.velocity, "channel": msg.channel},
      "time": time.time(),
    }
    print(time.time(),msg)
    await ws.send(json.dumps(fullMessage))

  # check if incoming ws message
  wsMessage = None
  try:
    wsMessage = await asyncio.wait_for(ws.recv(), timeout=0.001)
  except asyncio.TimeoutError:
    pass
  if wsMessage:
    notes = json.loads(wsMessage)
    curTime = time.time()
    ticksPerSecond = 500 * 2
    for note in notes:
      midoNote = mido.Message(note[0], channel=note[2], note=note[3], velocity=note[4])
      # dueTime = curTime + (note[1] / ticksPerSecond)
      dueTime = note[1]
      # insert sorted by due time
      msglog.append({"msg": midoNote, "due": dueTime})
      # curTime = dueTime
  # sort msglog by due time, since it's not guaranteed to be in order
  msglog = deque(sorted(msglog, key=lambda x: x["due"]))
  while len(msglog) > 0 and msglog[0]["due"] <= time.time():
    msg = msglog.popleft()["msg"]
    outport0.send(msg)
