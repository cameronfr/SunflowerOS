// MIDI

const onMIDISuccess = midiAccess => {
  midi = midiAccess;
  initDevices(midi)
}
const onMIDIFailure = msg => {
  console.error(`Failed to get MIDI access - ${msg}`);
}
navigator.requestMIDIAccess().then(onMIDISuccess, onMIDIFailure);

const initDevices = midi => {
  // Reset.
  midiIn = [];
  midiOut = [];
  // MIDI devices that send you data.
  const inputs = midi.inputs.values();
  for (let input = inputs.next(); input && !input.done; input = inputs.next()) {
    midiIn.push(input.value);
  }
  // MIDI devices that you send data to.
  const outputs = midi.outputs.values();
  for (let output = outputs.next(); output && !output.done; output = outputs.next()) {
    midiOut.push(output.value);
  }

  for (const input of midiIn) {
    input.addEventListener('midimessage', midiMessageReceived);
  }
}

const midiNoteToCharNote = midiNum => {
  const note = midiNum % 12;
  const octave = Math.floor(midiNum / 12) - 1;
  const noteLetter = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][note];
  return `${noteLetter}${octave}`;
}

// playHistory = []
noteOnEvents = {}
// onMidiMessage = (n,c, t) => {}
midiHistory = []
const midiMessageReceived = event => {
  // console.log(event.data);
  var [command, note, velocity] = event.data;
  command = command >> 4

  if (command === 9) {
    noteOnEvents[note] = Date.now()//event.timeStamp
  } else if (command === 8) {
    // Note Off event
    const noteOnTime = noteOnEvents[note]
    const duration = Date.now() - noteOnTime//event.timeStamp - noteOnTime
    midiHistory.push({note, time: noteOnTime, duration})
    delete noteOnEvents[note]
  }
}

// DRAWING
canvas = document.createElement('canvas')
canvas.width = document.body.clientWidth
canvas.height = document.body.clientHeight
document.body.appendChild(canvas)
ctx = canvas.getContext('2d')
// use high dpi
const dpi = window.devicePixelRatio
canvas.style.width = canvas.width + 'px'
canvas.style.height = canvas.height + 'px'
canvas.width *= dpi
canvas.height *= dpi
ctx.scale(dpi, dpi)

const screenWidth = document.body.clientWidth
const screenHeight = document.body.clientHeight
const numOctaves = 4
const numKeys = numOctaves * 12
const keyWidth = screenWidth / numKeys
var blackKeyWidth = 0.8
var whiteKeyWidth = 1
var octaveWidth = blackKeyWidth * 5 + whiteKeyWidth * 7
blackKeyWidth *= screenWidth / (octaveWidth * numOctaves)
whiteKeyWidth *= screenWidth / (octaveWidth * numOctaves)

const noteColors = ["#8AE36C", "#BE88A2", "#FBD061", "#FD924B", "#E8EB3D", "#70E7F6", "#587dc2", "#FF9F16", "#B7B4A5", "#FF3942", "#786D67", "#FD8699"]
const startKeyMidi = 60-24
const scrollSpeed = 0.05//0.01//0.05


var drawBackground = () => {
  for (let i = 0; i < numKeys; i++) {
    ctx.fillStyle = i % 12 === 1 || i % 12 === 3 || i % 12 === 6 || i % 12 === 8 || i % 12 === 10 ? '#eee' : 'white'
    // ctx.fillStyle = noteColors[i % 12]
    ctx.fillRect(i * keyWidth, 0, keyWidth, screenHeight)
  }
  // draw line at octave boundaries
  for (let i = 0; i < numOctaves; i++) {
    ctx.fillStyle = '#aaa'
    ctx.fillRect(i * keyWidth * 12, 0, 1, screenHeight)
  }

  // add horizontal lines every 17px / scrollSpeed
  for (let i = 0; i < screenHeight; i += 500.1 * scrollSpeed) {
    ctx.fillStyle = '#ccc'
    ctx.fillRect(0, i, screenWidth, 0.5)
  }

}

//start vertical scrolling animation
let scrollY = 0
const animate = () => {
  ctx.clearRect(0, 0, screenWidth, screenHeight)
  drawBackground()

  var drawNote = (note, time, duration) => {
    const noteX = (note - startKeyMidi) * keyWidth
    const timeAgo = Date.now() - time
    const noteY = (scrollY - timeAgo) * scrollSpeed + screenHeight
    const noteHeight = duration * scrollSpeed
    ctx.fillStyle = noteColors[note % 12]
    // add stroke
    ctx.strokeStyle = 'black'
    ctx.lineWidth = 1
    ctx.strokeRect(noteX, noteY, keyWidth, noteHeight)
    ctx.fillRect(noteX, noteY, keyWidth, noteHeight)
  }

  // draw old notes
  for (const {note, time, duration} of midiHistory) {
    drawNote(note, time, duration)
  }
  // draw current notes (that aren't off yet)
  for (const note in noteOnEvents) {
    drawNote(note, noteOnEvents[note], Date.now() - noteOnEvents[note])
  }

  requestAnimationFrame(animate)
}
animate()


onMidiMessage = (n, c, t) => {
  console.log(n, t)
  n
  c
}