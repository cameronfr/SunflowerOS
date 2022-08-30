import { printLine } from './modules/print';
import React from 'react';
import * as ReactDOM from 'react-dom';

console.log('Content script works! 4');
console.log('Must reload extension for modifications to take effect.');

const loadingSpinner = <>
    <svg aria-hidden="true" class="w-4 h-4 text-gray-200 animate-spin dark:text-gray-600 fill-blue-600" viewBox="0 0 100 101" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z" fill="currentColor"/>
        <path d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z" fill="currentFill"/>
    </svg>
</>

var body = document.querySelector("body")
// Append script tag to head with src https://cdn.tailwindcss.com
// var script = document.createElement("script")
// script.src = "https://cdn.tailwindcss.com"
// script.onload = () => {
//     console.log("Tailwind loaded")
// }
// document.querySelector("head").appendChild(script)

var modifyTweets = () => {
    body.removeEventListener('DOMSubtreeModified', modifyTweets);
    // Append button to every tweet on the page
    const tweets = document.querySelectorAll('article');
    tweets.forEach((tweet) => {
        if (tweet.querySelector(':scope > .pantheon-modified-tweet')) {
            return;
        }
        var div = document.createElement('div');
        div.style.display = 'none';
        div.classList.add("pantheon-modified-tweet")
        tweet.appendChild(div)


        // Get tweet link from node
        var tweetDirectLinkRel = Array.from(tweet.querySelectorAll("a")).filter(a => {
            // regex to check if a.href contains /status/<number>
            var isId = /\/status\/\d+$/.test(a.href)
            return isId
        })[0]
        var tweetDirectLink = new URL(tweetDirectLinkRel, document.baseURI).href

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

            var onClick = async () => {
                if (loading) {
                    // don't do anything if already loading
                    return
                }
                // Send post request to localhost:3000 with tweet link as tweetURL, using fetch
                console.log("Fetching for link", tweetDirectLink)
                setLoading(true)
                var res = await fetch("http://localhost:8089/generate", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        tweetURL: tweetDirectLink
                    })
                })
                var generation = await res.json()
                setGeneration(generation)
                setLoading(false)
                // setGeneration({classification: "excitement-fomo", thoughtGuide: "Don't worry about the hyped thing, you'll know when to use it! Stay focused on what you're working on and you'll be fine."})
            }

            const tweetTextContainer = tweet.querySelector("div[data-testid='tweetText']")
            const authorBar = tweet.querySelector("div[data-testid='User-Names']")?.parentNode?.parentNode?.parentNode

            var textAppend = <>
                <span className="pl-1 text-teal-600">
                    {generation?.thoughtGuide} 
                    <span style={{fontSize: "0.5em"}} className="pl-1 pr-1 bg-teal-600 text-white rounded-sm">
                        {generation?.classification}
                    </span>
                </span>
            </>

            return <>
                <InjectComponentAtDomNode domNode={tweetTextContainer} containerType={"span"}>
                    {generation && textAppend}
                </InjectComponentAtDomNode>
                <InjectComponentAtDomNode domNode={authorBar} insertBeforeNode={authorBar?.children?.[1]} containerType={"div"} containerClassnames="grow flex justify-end">
                    <div>
                        <button className="flex justify-center items-center pl-1 pr-1 rounded-sm bg-sky-400 hover:bg-sky-500 text-white" onClick={onClick}>
                            <span>guide</span>
                            {loading && <div className="ml-2" style={{height: "1em"}}>{loadingSpinner}</div>}
                        </button>
                    </div>
                </InjectComponentAtDomNode>
            </>
        }
        var mainContainer = document.createElement("div")
        ReactDOM.render(<MainComponent />, mainContainer)
        document.body.appendChild(mainContainer)
    });

    body.addEventListener("DOMSubtreeModified", modifyTweets);
}

body.addEventListener("DOMSubtreeModified", modifyTweets);
