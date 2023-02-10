import React from "react"
import ReactDOM from "react-dom"
import CodeMirror from '@uiw/react-codemirror';
import localforage from "localforage"

import { Configuration, OpenAIApi } from "openai"

const configuration = new Configuration({
  apiKey: "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",
});
const openai = new OpenAIApi(configuration);
/*
const response = openai.createCompletion({
  model: "text-davinci-003",
  prompt: "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: ",
  temperature: 0.9,
  max_tokens: 150,
  top_p: 1,
  frequency_penalty: 0,
  presence_penalty: 0.6,
  stop: [" Human:", " AI:"],
}).then(t => {
  t
}
)*/

var MainComponent = () => {
  // Load saved state
  var [didInitialLoad, setDidInitialLoad] = React.useState(false)
  var [initialCode, setInitialCode] = React.useState("default")  
  React.useEffect(() => {
    localforage.getItem("code").then((value) => {
      if (value) {
        setInitialCode(value)
      }
      setDidInitialLoad(true)
    })
  }, [])

  var [curText, setCurText] = React.useState("")
  var lastSentText = React.useRef()

  var analyzeText = text => {
    text
  }

  // If no activity for 5s & text changed from last trigger, trigger something
  React.useEffect(() => {
    var interval = setTimeout(() => {
      console.log("5 seconds")
      if (lastSentText == curText) {return}
      lastSentText = curText
      analyzeText(curText)
    }, 5000)
    return () => {
      clearInterval(interval)
    }
  })

  var out = <>
    <CodeMirror
      value={initialCode}
      options={{
      }}
      onUpdate={instance => {
        // Set new saved state. instance is CodeMirror.Editor instance
        const newText = instance.view.state.doc.toString()
        didInitialLoad && localforage.setItem("code", newText)
        setCurText(newText)
      }}
    /> 
  </>
  return out
}

root = ReactDOM.createRoot(document.querySelector("body"))
root.render(<MainComponent/>)