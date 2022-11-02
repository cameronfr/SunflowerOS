// Import the module and reference it with the alias vscode in your code below
import { json } from 'stream/consumers';
import * as vscode from 'vscode';
import * as zmq from 'zeromq';

var sock;
// {[filename]: {[pos]: decoration}}
var currentDecorations = {}

// This obj instance is used as key for setDecorations -- if change every time, won't remove old 
const decorationType = vscode.window.createTextEditorDecorationType({
  backgroundColor: 'green',
  border: '2px solid white',
})

export function activate(context: vscode.ExtensionContext) {
  console.log('Activating sunflowereditor extension')

  // Custom timeout is best for now, so don't have to think about low level socket stuff
  // Also, if want to bind (instead of connect here), need to have ssh remote->local tunnel, which would be annoying.
  var sockTimeout = 5000
  var sockTimeoutObj = null

  var recreateSocket = () => {
    if (sock) {
      sock.close()
      console.log("ZMQ socket timed out, recreating")
    }
    sock = zmq.socket('sub')
    sock.connect('tcp://0.0.0.0:5895')
    sock.subscribe('')
    console.log("Started ZMQ subscribe")
    sock.on('message', onSockMessage)
    sockTimeoutObj = setTimeout(recreateSocket, sockTimeout)
  }

  var onSockMessage = (topic, msg) => {
    console.log(`Received message: [${topic}] ${msg}`)
    clearTimeout(sockTimeoutObj)
    sockTimeoutObj = setTimeout(recreateSocket, sockTimeout)
    if (topic == "heartbeat") {
      return
    } 

    let msgJSON = JSON.parse(msg.toString())
    parseMessage(msgJSON)
    if (vscode.window.activeTextEditor) {
      drawDecorations(vscode.window.activeTextEditor)
    }

  }

  recreateSocket()

  // on window change, draw decorations
  vscode.window.onDidChangeActiveTextEditor(editor => {
    console.log("window changed")
    if (editor) {
      drawDecorations(editor)
    }
  }, null, context.subscriptions);

  // try {
  //   parseMessage({
  //     filepath: "/test/main_hot.cpp",
  //     line: 10,
  //     lineChar: 4,
  //     text: "hello",
  //     timestamp: new Date().getTime(),
  //   })
  //   drawDecorations(vscode.window.activeTextEditor)
  // } catch (e) {
  //   console.error(e.stack)
  // }
}

var drawDecorations = (editor: vscode.TextEditor) => {

  let decorations = []
  let filepath = editor.document.fileName
  let filename = filepath.split("/").pop()
  let decorationsForFile = currentDecorations[filename]
  if (decorationsForFile) {
    for (let posKey in decorationsForFile) {
      const vscodeDecoration = decorationsForFile[posKey].vscodeDecoration
      decorations.push(vscodeDecoration)
    }
  }

  console.log("decorations len is " + decorations.length)
  console.log("decorations is ", JSON.stringify(decorations))

  editor.setDecorations(decorationType, decorations)
}

var parseMessage = (msgJSON) => {
  const filepath = msgJSON.filepath
  const text = msgJSON.text
  const line = msgJSON.line
  const timestamp = msgJSON.timestamp
  const lineChar = msgJSON.lineChar

  const fileKey = filepath.split("/").pop()
  const posKey = line + ":" + lineChar

  const range = new vscode.Range(
    new vscode.Position(line, lineChar), 
    new vscode.Position(line, lineChar)
  )

  // Time in form e.g. 5:45:30
  let timeString = new Date(timestamp).toLocaleTimeString('en-US', { hour12: true, hour: "numeric", minute: "numeric", second: "numeric" })
  // https://github.com/microsoft/vscode/blob/6d2920473c6f13759c978dd89104c4270a83422d/src/vs/base/browser/markdownRenderer.ts#L296 //allowed tags and attribute and style string. Note style string sanitization is picky.
  let hoverMessage = new vscode.MarkdownString(`
    <strong><span style="color:#af005f;">Time</span>:</strong> ${timeString}
  `)
  // remove indentation so we can indent the code block above. if don't, html tags stop working 
  hoverMessage.value = hoverMessage.value.replace(/^\s+/gm, '')
  hoverMessage.isTrusted = true
  hoverMessage.supportHtml = true
  var vscodeDecoration : vscode.DecorationOptions = {
    range: range,
    hoverMessage: hoverMessage,
    renderOptions: {
      after: {
        contentText: text,
        color: "gray",
      }
    },
  }

  var decoration  = {
    vscodeDecoration
  }

  currentDecorations[fileKey] = currentDecorations[filepath] || {}
  currentDecorations[fileKey][posKey] = decoration
}

// This method is called when your extension is deactivated
export function deactivate() {}
