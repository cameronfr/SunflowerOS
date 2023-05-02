import { printLine } from './modules/print';
import React, { useEffect } from 'react';
import * as ReactDOM from 'react-dom';

console.log('Content script works! 6');
console.log('Must reload extension for modifications to take effect.');

var totalTokensUsed = 0

const loadingSpinner = <>
  <svg aria-hidden="true" class="w-4 h-4 text-gray-200 animate-spin dark:text-gray-600 fill-white" viewBox="0 0 100 101" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z" fill="currentColor"/>
      <path d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z" fill="currentFill"/>
  </svg>
</>


var modifyTextWithModel = async (text) => {
  const OPENAI_KEY = "<YOUR_KEY_HERE, e.g. sk-... >"
  const OPENAI_URL = "https://api.openai.com/v1/chat/completions"
  // const instruction = `Please rewrite the following text, making it as if it were a short text message from someone who loves and cares about me. Assume we have already greeted each other, so omit any greeting. Please try and keep the rewritten text about the same length or shorter the original text:\n\nBEGIN_INPUT_TEXT ${text} END_INPUT_TEXT`
  // const instruction = `Please rewrite the following text to be like a short message from someone who acts in the following way: they love and care about me, they talk in lowercase, and they write tersely but kindly. Assume we have already greeted each other, so omit any greeting. Please try and keep the rewritten text about the same length or shorter the original text. Don't change the meaning of the text.:\n\nBEGIN_INPUT_TEXT ${text} END_INPUT_TEXT`
  const instruction = `Please rewrite the following message to be like a message from someone who is very kind and loving, and talks in very cutely in lowercase. Change the message as little as possible, don't change it's meaning, and don't add fluff. Also, if the message is overhyping something, make it more down to earth.\nBEGIN_INPUT_TEXT\n${text}\nEND_INPUT_TEXT`
  //Also, if the message is in another language, translate it to English first
  //Also, if the message is overhyping something, make it more down to earth.

  var messages = [
    {"role": "user", "content": instruction},
  ]

  var res = await fetch(OPENAI_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${OPENAI_KEY}`
    },
    body: JSON.stringify({
      model: "gpt-3.5-turbo",
      temperature: 0.7,
      messages
    })
  }).then(res => res.json())
  totalTokensUsed += res.usage.total_tokens
  console.log("total tokens used: " + totalTokensUsed, ", total cost: $" + (0.002 * totalTokensUsed / 1000.0))

  var messageOut = res.choices[0].message
  return messageOut.content
}

var body = document.querySelector("body")

var modifyTweets = () => {
  body.removeEventListener('DOMSubtreeModified', modifyTweets);
  // Append button to every tweet on the page
  // const tweets = document.querySelectorAll('article');
  const tweets = [...document.querySelectorAll("div[data-testid='tweetText']")].map(t => t.parentElement)
  tweets.forEach((tweet) => {
    if (tweet.querySelector(':scope > .pantheon-modified-tweet')) {
        return;
    }
    var div = document.createElement('div');
    div.style.display = 'none';
    div.classList.add("pantheon-modified-tweet")
    tweet.appendChild(div)

    var InjectComponentAtDomNode = ({domNode, children, containerType, containerClassnames, insertBeforeNode}) => {
        // TODO: if insertBeforeNode / domNode changes, not currently handled
        var [rootNode, setRootNode] = React.useState(() => {
          var container = document.createElement(containerType)
          container.className = containerClassnames || ""
          if (domNode) {
            if (insertBeforeNode) {
              domNode.insertBefore(container, insertBeforeNode)
            } else {
              domNode.appendChild(container)
            }
          } else {
            console.log("ERROR: didn't find domnode to insert at")
          }
          return container
        })

        React.useEffect(() => {
          ReactDOM.render(<>{children}</>, rootNode)
        })

        return null
    }

    var MainComponent = () => {
      var [loading, setLoading] = React.useState(false)
      var [generation, setGeneration] = React.useState()
      var originalTextRef = React.useRef()

      var onClick = async () => {
          // setGeneration({classification: "excitement-fomo", thoughtGuide: "Don't worry about the hyped thing, you'll know when to use it! Stay focused on what you're working on and you'll be fine."}); return
          // setLoading(true)
          // var generation = await res.json()
          // console.log(generation)
          // setGeneration(generation)
          // setLoading(false)
      }

      const tweetTextContainer = tweet.querySelector("div[data-testid='tweetText']")

      useEffect(() => {
        if (tweetTextContainer) {
          var text = tweetTextContainer.innerText
          var originalOpacity = tweetTextContainer.style.opacity
          tweetTextContainer.style.opacity = 0.0
          originalTextRef.current = text
          modifyTextWithModel(text).then((newText) => {
            tweetTextContainer.innerText = newText
          tweetTextContainer.style.opacity = originalOpacity
          })
        }
      }, [tweetTextContainer])

      return null

      
      // <InjectComponentAtDomNode domNode={tweetTextContainer} /*insertBeforeNode={authorBar?.children?.[1]}*/ containerType={"div"} containerClassnames="grow flex justify-end">
      //     <div>
      //         <button className="flex justify-center items-center pl-1 pr-1 rounded-sm bg-sky-400 hover:bg-sky-500 text-white" onClick={onClick}>
      //             <span>guide</span>
      //             {loading && <div className="ml-2" style={{height: "1em"}}>{loadingSpinner}</div>}
      //         </button>
      //     </div>
      // </InjectComponentAtDomNode>
    }
    var mainContainer = document.createElement("div")
    ReactDOM.render(<MainComponent />, mainContainer)
    document.body.appendChild(mainContainer)
  });

  body.addEventListener("DOMSubtreeModified", modifyTweets);
}

body.addEventListener("DOMSubtreeModified", modifyTweets);
