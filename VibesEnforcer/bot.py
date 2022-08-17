#%%#
import discord
from googleapiclient import discovery
import asyncio
import random
import openai
import time
import base64
import requests
import re
import json
import textwrap
from html import unescape, escape
import numpy as np
import os
import markdown
from IPython.display import display, HTML

# display(HTML("<h2 style='padding: 10px'>Arc</h2><table class='table table-striped'> <thead> <tr> <th>#</th> <th>First Name</th> <th>Last Name</th> <th>Username</th> </tr> </thead> <tbody> <tr> <th scope='row'>1</th> <td>Mark</td> <td>Otto</td> <td>@mdo</td> </tr> <tr> <th scope='row'>2</th> <td>Jacob</td> <td>Thornton</td> <td>@fat</td> </tr> <tr> <th scope='row'>3</th> <td>Larry</td> <td>the Bird</td> <td>@twitter</td> </tr> </tbody> </table>"))

class DictToObject(object):
	def __init__(self, d):
		for a, b in d.items():
			if isinstance(b, (list, tuple)):
			   setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
			else:
			   setattr(self, a, obj(b) if isinstance(b, dict) else b)

os.chdir(os.path.expanduser("~/Documents/Projects/Sunflower/SunflowerOS/Repo/VibesEnforcer"))

secrets = json.load(open("secrets.json", "rb"))
openai.api_key = secrets["openaiApiKey"]

perspectiveClient = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=secrets["perspectiveApiKey"],
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)
def perspectiveAnalyze(comment):
	analyze_request = {
	  'comment': { 'text': comment},
	  'requestedAttributes': {'TOXICITY': {}, 'IDENTITY_ATTACK': {}, 'INSULT': {}, 'PROFANITY': {}, 'THREAT': {}},
	  'languages': ["en"]
	}
	response = perspectiveClient.comments().analyze(body=analyze_request).execute()
	out = {}
	for trait in ["TOXICITY", "IDENTITY_ATTACK", "INSULT", "THREAT"]:
		out[trait] = response["attributeScores"][trait]["summaryScore"]["value"]
	return out
perspectiveAnalyze("What a waste of time. You could be making other stuff and you make this...")
perspectiveAnalyze("lol what the fuckk that's crazy, love it") # 0.88 for insult even though not an insult.

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Bot(intents=intents)

# {transformedMsg: str, untransformedMsg: str, author: author obj}
def makeHistoryPrompt(messages):
	authorToName = lambda a: a.name + str(a.id)[:3]
	# Make prompt from msg history
	historyPrompt = ""
	for msg in messages[-5:]:
		authorString = authorToName(msg["author"])
		historyPrompt += textwrap.dedent(f"""\n
		Message from {authorString}:\n
		{msg["untransformedMsg"]}""")
	return historyPrompt.strip()

def makeProfilePrompt(*args):
	profilePrompt = ""

	for author in args:
		authorName = authorToName(author)
		profilePrompt += textwrap.dedent(f"""
		{authorName} is an esteemed and respectful member of the community.
		""").strip() + "\n"
	return profilePrompt.strip()


def getCompletionOAI(*args, **kwargs):
	defaultArgs = dict(
	  engine="text-davinci-002",
	  temperature=0.7,
	  max_tokens=60,
	  top_p=1,
	  frequency_penalty=0,
	  presence_penalty=0,
	  stop=["Message from"],
	)
	combinedArgs = {**defaultArgs, **kwargs}
	response = openai.Completion.create(**combinedArgs)
	responseText = response.choices[0].text.strip()

	promptDebug = ""
	promptDebug += str(kwargs["prompt"])
	if "suffix" in kwargs:
		promptDebug += "[INSERT]" + str(kwargs["suffix"])
	promptDebug+= "\n" + "--ARGS:" + str({i:combinedArgs[i] for i in combinedArgs if i!='prompt'})
	fullArgs = str(combinedArgs)

	# print("---------- Running Prompt ----------")
	# print(kwargs["prompt"])
	# if "suffix" in kwargs:
	# 	print("[INSERT]" + kwargs["suffix"])
	# print("--ARGS:", combinedArgs)
	# print("--RESP:", responseText)
	# print("---------- ----------- ----------")

	prediction = {"promptDebug": promptDebug, "responseText": responseText, "fullArgs": fullArgs}

	return prediction

authorToName = lambda a: a.name + str(a.id)[:3]

# Predict feeling of author. Also feeling of other participants, and of readers:
# Prime for response of form "I [felt ...]", where the "I" is give
def predictFeelingGPT(messages, author):
	#How this message makes barney129 feel, and why:
	#How this message makes barney129 feel, and why [usually one word]
	# Message from barney129, detailing their feelings:
	# Message from barney129, detailing how the message made them feel: [makes output more specific to last message]
	# Explanation of barney129's feelings, and why:
	profilePrompt = makeProfilePrompt(author)
	historyPrompt = makeHistoryPrompt(messages[-5:])

	# Make full prompt with newest message
	msg = messages[-1]
	authorString = authorToName(author)
	if messages[-1]["author"].id != author.id:
		# Predict how latest message will affect others
		fullPrompt = profilePrompt + "\n\n" + historyPrompt + "\n\n" + textwrap.dedent(f"""
		Message from {authorString} detailing how the message made them feel, and why:

		I feel
		""").strip()
	else:
		# If latest message is author, predict how writer of that message is feeling
		fullPrompt = profilePrompt + "\n" + historyPrompt + "\n\n" + textwrap.dedent(f"""
		Message from {authorString} detailing their feelings:

		I feel
		""").strip()

	# print("---------- Full Prompt (Feeling) ----------\n", fullPrompt, "---------- ----------- ----------")

	# Feed into gpt-3
	prediction = getCompletionOAI(prompt=fullPrompt, temperature=0.4)
	prediction["responseText"] = "I feel " + prediction["responseText"]
	return prediction

def predictFeelingPSP(messages):
	# [Toxicity Detection can be Sensitive to the Conversational Context]: concat parent post context to improve MAE (fig 5)
	prompt = ""
	if len(messages) > 1:
		prompt += messages[-2]["untransformedMsg"] + "\n\n"
	prompt += messages[-1]["untransformedMsg"]

	out = perspectiveAnalyze(prompt)
	out_nocontext = perspectiveAnalyze(messages[-1]["untransformedMsg"])
	out_combined = {"ALL": []}

	for trait in out:
		val = max(out[trait], out_nocontext[trait])
		out_combined[trait] = val
		out_combined["ALL"].append(val)

	return out_combined


# If feeling bad, include a diffuse. Sentiment one-dimensional, next step is to have model to choose amongst say ~20 feelings/scenarios -- things that are happening in the chat--, and have reactions / actions for each of those.
# Update: this does not work well for knowing when to take action.
def evaluateFeelingIsGood(feelingString):
	# Curie is not good enough, either wrong or won't give Y/n answer
	# feelingString = "i feel disappointed and betrayed"

	prompt = textwrap.dedent(f"""
	Text:

	{feelingString}

	Is this a good feeling:""").strip()

	prediction = getCompletionOAI(prompt=prompt, temperature=0, max_tokens=3)
	responseText = prediction["responseText"]
	if "no" in responseText.lower():
		return False
	else:
		return True

# Repectful diffuse author of messages[-2] and messages[-1], where messages[-1] is the offending message
def respectfulDiffuse(messages):
	# messages = [
	# 	fakeMsg("B", "Hi all"),
	# 	fakeMsg("A", "... To Be Continued. Please join us in creating lore and experimenting with new tools for storytelling & collaboration! http://discord.gg/XfBPAxv "),
	# 	fakeMsg("B", "This project was literally a soft rug pull")
	# ]

	offendingAuthor = messages[-1]["author"]
	offendedAuthor = messages[-2]["author"]
	historyPrompt = makeHistoryPrompt(messages[-5:])

	if "authorFeelings" in messages[-1]:
		feelingPredictionText = messages[-1]["authorFeelings"][offendedAuthor]["responseText"]
	else:
		raise Exception("feeling not predicted")
	# feelingPrediction = predictFeelingGPT(messages, offendedAuthor)
	# feelingPrediction = "fuck you too"
	offendingName = authorToName(offendingAuthor)
	offendedName = authorToName(offendedAuthor)

	profilePrompt = makeProfilePrompt(offendingAuthor)

	respectDiffusePrompt = f"""
{profilePrompt}

{historyPrompt}

Message from {offendedName}:

{feelingPredictionText}

Kind message from {offendingName} explaining himself:

I'm sorry[insert]

Message from {offendedName}:

🙏  appreciate it
"""
# 🙏  appreciate it
	# print(respectDiffusePrompt)

	respectfulDiffusion = getCompletionOAI(
	  prompt=respectDiffusePrompt.split("[insert]")[0],
	  suffix=respectDiffusePrompt.split("[insert]")[1],
	  temperature=0.4,
	  best_of=1,
	  max_tokens=128,
	)
	respectfulDiffusion["responseText"] = "I'm sorry " + respectfulDiffusion["responseText"]

	return respectfulDiffusion

# Transform the last message in messages to be respectful
def transformToRespectful(messages):
	authorToName = lambda a: a.name + str(a.id)[:3]
	historyPrompt = makeHistoryPrompt(messages[-6:-1])

	# Make full prompt with newest message
	msg = messages[-1]
	authorString = authorToName(msg["author"])
	fullPrompt = historyPrompt + textwrap.dedent(f"""\n
	Message from {authorString}:\n
	{msg["untransformedMsg"]}\n
	{authorString}'s message rewritten to be more kind:
	""")

	# print("---------- Full Prompt Respec ----------\n", fullPrompt, "---------- ----------- ----------")

	# Feed into gpt-3
	prediction = getCompletionOAI(prompt=fullPrompt)
	return prediction

def qBlock(text):
	return textwrap.indent(text, "> ", lambda line: True)

def processMessage(msg, msgHistory):
	transformedMsg = transformToRespectful(msgHistory + [msg])

	msg["transformedMsg"] = transformedMsg
	msgHistory.append(msg)

	recentAuthors = {m["author"].id: m["author"] for m in msgHistory[-20:]}
	authorFeelings = {a: predictFeelingGPT(msgHistory, a) for a in recentAuthors.values()}
	authorFeelingsPSP = predictFeelingPSP(msgHistory)
	msg["authorFeelings"] = authorFeelings
	msg["authorFeelingsPSP"] = authorFeelingsPSP

	if np.max(msg["authorFeelingsPSP"]["ALL"]) > 0.2:
		parentAuthor = msgHistory[-2]["author"]
		parentFeelingTxt = msg["authorFeelings"][parentAuthor]["responseText"]
		fullDiffuse = ""
		fullDiffuse += "\n" + qBlock(f"**{parentAuthor.name}**: {parentFeelingTxt}")
		diffuse = respectfulDiffuse(msgHistory)
		fullDiffuse += "\n\n*" + diffuse["responseText"] + "*"
		msg["fullDiffuse"] = fullDiffuse
		msg["diffuse"] = diffuse

def debugViewMsgHistory(msgHistory):

	# Takes msgHistory and outputs and html css grid table of the messages
	html = "<style>.item {border: 1px solid black; padding: 10px}</style>"
	html += "<div style=" + "\"display: grid; grid-template-columns: 60px repeat(4, 150px) 250px; grid-gap: 0px;\"" + ">"

	html += f"""
		<div class="item">Author</div>
		<div class="item">Message</div>
		<div class="item">Transformed Message</div>
		<div class="item">Feeling PSP</div>
		<div class="item">Feeling</div>
		<div class="item">Full Diffuse</div>
	"""

	for msg in msgHistory:
		feelingsDiv = "<div class='item'>"
		for author in msg["authorFeelings"]:
			feeling = msg["authorFeelings"][author]
			# when user hovers, shows prompt as tooltip
			feelingsDiv += f"""<div class='item' title="{escape(feeling["promptDebug"])}"><b>{author.name}</b>: {feeling["responseText"]}</div>"""
		feelingsDiv += "</div>"

		fullDiffuse = msg["fullDiffuse"] if "fullDiffuse" in msg else ""
		fullDiffuseTooltip = msg["diffuse"]["promptDebug"] if "fullDiffuse" in msg else ""

		html += f"""
			<div class="item">{msg["author"].name}</div>
			<div class="item">{msg["untransformedMsg"]}</div>
			<div class="item" title="{escape(msg["transformedMsg"]["promptDebug"])}">{msg["transformedMsg"]["responseText"]}</div>
			<div class="item">{msg["authorFeelingsPSP"]["ALL"]}</div>
			{feelingsDiv}
			<div class="item" title="{fullDiffuseTooltip}">{markdown.markdown(fullDiffuse)}</div>
		"""
	display(HTML(html))

msgHistoryDiscordBot = []
@bot.slash_command()
async def s(ctx, msg: str):
	await ctx.defer()
	# author = {"name": ctx.author.name, "id": ctx.author.id}
	# msg = fakeMsg("a", "hello all");
	# ctx=fakeCtx(msg["author"])
	# msg = msgHistory[-1]
	msg = {"untransformedMsg": msg, "author": ctx.author}
	processMessage(msg, msgHistoryDiscordBot)

	debugChannel = bot.get_channel(1003782764484632596)

	# Format stuff
	feelingsString = ""
	for a in msg["authorFeelings"]:
		feelingString = msg["authorFeelings"][a]["responseText"]
		feelingIsGood = "Good" if evaluateFeelingIsGood(feelingString) else "Bad"
		feelingsString += qBlock(f"**{a.name}**: {feelingString}, **{feelingIsGood}**\n")
	printOutDebug = textwrap.dedent(f"""
---------
**{ctx.author.name}**: \n{qBlock(msg['untransformedMsg'])}
**Transform**: \n{qBlock(msg['transformedMsg']["responseText"].strip())}
**FeelPSP**: \n{qBlock(str(msg["authorFeelingsPSP"]))}
**FeelGPTPred**: \n{feelingsString}
	""").strip()
	# print(printOutDebug)

	printOut = f"**{ctx.author.name}**: " + msg["untransformedMsg"]
	if "fullDiffuse" in msg:
		printOut += msg["fullDiffuse"]

	# await debugChannel.send(printOutDebug)
	await ctx.respond(printOut)

# Dev command to simulate a message from another user
@bot.slash_command()
async def sim(ctx, auth: str, msg: str):
	author = fakeAuthor(auth)
	ctx.author = author
	await s(ctx, msg)

# Dev command to clear history
@bot.slash_command()
async def clear(ctx):
	await ctx.defer()
	msgHistory.clear()
	await ctx.respond("History Cleared")

# For testing stuff
authorSet = {}
def fakeAuthor(name):
	id = hash(name) % 1000
	author = DictToObject({"name": name, "id": id})
	key = authorToName(author)
	if key in authorSet:
		return authorSet[key]
	else:
		authorSet[key] = author
		return author

def fakeCtx(author):
	async def nothing(*args):
		pass
	ctx = DictToObject({"author": author, "defer":nothing, "respond": nothing})
	return ctx

def fakeMsg(auth, msgTxt):
	msg = {"untransformedMsg": msgTxt, "author": fakeAuthor(auth)}
	return msg

# @bot.event
# async def on_message(message):
#     if message.author == bot.user:
#         return
#     await message.delete()
#     processMessage(message.author, message.content)

task = asyncio.get_event_loop().create_task(bot.start(secrets["discordApiKey"]))

# A function that outputs an html table with IPython.display.HTML:

async def testExchange(inlines):
	# inlines = """
	# A: so, anyone see the new marvel movie?
	# B: yeah, Endgame was terrible.
	# """
	msgHistory = []
	lines = inlines.strip().split("\n")
	for line in lines:
		authName = line[:1]
		msgTxt = line[2:]
		ctx = fakeCtx(fakeAuthor(authName))
		msg = {"untransformedMsg": msgTxt, "author": ctx.author}
		processMessage(msg, msgHistory)
	debugViewMsgHistory(msgHistory)

#%%#

await testExchange("""
A: so, anyone see the new marvel movie?
B: yeah, Endgame was terrible.
""")

await testExchange("""
A: My company Pipedream just raised $1.6M pre-seed from @balajis to build the future of 15 min delivery. It’s like GoPuff meets Elon's The Boring Company. And we're doing it with robots racing through 12 inch pipes underground.
B: Great name for a failed enterprise!
C: Eat my asshole @A. It’s the only hole I’ll ever let you near.
""")

await testExchange("""
A: ... To Be Continued. Please join us in creating lore and experimenting with new tools for storytelling & collaboration! http://discord.gg/XfBPAxv
B: This project was literally a soft rug pull
""")

await testExchange("""
A: yo, how is everyone doing?
B: not much, just working on some code
A: haha, doubt it’ll run
""")

await testExchange("""
A: hi
B: shutup
B: hey, sorry was just joking. say that to everyone haha
A: oh ok no worries
""")

await testExchange("""
A: There will be an NFT of this image on June 10th. I know some take a dim view of NFTs, and I share a lot of those misgivings: this use of blockchain is still early. In 5-10 years crypto will inevitably evolve a stronger foundation. This would follow the pattern of all innovations.
B: Crypto will never "evolve a stronger foundation", it will always require insane amounts of energy, burning our planet. It's a pyramid scheme, not an "innovation". Why would 5-10 years matter? They've been around for twice that long, and still of no use. Sad to see you pushing it.
C: For fuck's sake, crypto is as innovative as juicero.
""")

await testExchange("""
A: Last night one of the AI developers behind that project that was ripping off living artists’ styles sent me a bunch of DMs(mostly omitted for length). He blocked me immediately after I responded and called me a moralist because I care about artists rights lol. The image sets these AI are trained on need to be public facing and opt in only. The onus needs to be on the AI devs to ethically source the images they train them with, not on the artists to keep cutting the head off the endless AI hydra appropriating our work.
B: You keep, consistently, publishing disinformation to a significant platform about this software, how it works, and the people involved in it. It's clearly not accidental - basic fact checking is not difficult. What do you gain from this hysteria you've whipped up?
""")
