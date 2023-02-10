import mido
import json
import time
import asyncio
import websockets
mido.set_backend('mido.backends.rtmidi')
from collections import deque
!pip show websocket-client

print(mido.get_input_names())
print(mido.get_output_names())

deviceName = "LUMI Keys BLOCK"
inport = mido.open_input("LUMI Keys BLOCK")
outportForAudio = mido.open_output("IAC Driver Bus 1")
outportForDiplay = mido.open_output("LUMI Keys BLOCK")

msglog = deque()
echo_delay = 0.1

msg = mido.Message('note_on', note=60, velocity=64, time=0)
outportForAudio.send(msg)
outportForDiplay.send(msg)

# connect to websocket at desktop-3vakahr:8765
# ws = websocket.WebSocket()
# ws.connect("ws://desktop-3vakahr:8889")
ws = await websockets.connect("ws://desktop-3vakahr:8889")

#Rewritten for new version of mido
while True:
  msg = inport.poll()
  if msg and msg.type in ["note_on", "note_off", "control_change"]:
    fullMessage = {
      "midi": {"type": msg.type, "note": msg.note, "velocity": msg.velocity, "channel": msg.channel},
      "time": time.time(),
    }
    await ws.send(json.dumps(fullMessage))

  # check if incoming ws message
  wsMessage = None
  try:
    wsMessage = await asyncio.wait_for(ws.recv(), timeout=0.01)
  except asyncio.TimeoutError:
    pass
  if wsMessage:
    notes = json.loads(wsMessage)
    curTime = time.time()
    ticksPerSecond = 500 * 2
    for note in notes:
      midoNote = mido.Message(note[0], channel=note[2], note=note[3], velocity=note[4])
      dueTime = curTime + (note[1] / ticksPerSecond)
      msglog.append({"msg": midoNote, "due": dueTime})
      curTime = dueTime
    # msglog.append({"msg": msg, "due": time.time() + echo_delay})
  while len(msglog) > 0 and msglog[0]["due"] <= time.time():
    msg = msglog.popleft()["msg"]
    outportForDiplay.send(msg)
    outportForAudio.send(msg)



