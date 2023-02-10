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

playHistory = []
const midiMessageReceived = event => {
  // console.log(event.data);
  var [command, note, velocity] = event.data;
  command = command >> 4
  if (command === 9 && velocity > 0) {
    const timeDesc = (event.timeStamp/1000).toFixed(2) 
    const noteDesc = midiNoteToCharNote(note)
    playHistory.push(`${noteDesc} at ${timeDesc}s`)
  }
  playHistory.join("\n")
}
