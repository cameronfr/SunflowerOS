#%%#
from cgitb import text
import discord
# from googleapiclient import discovery
from aiogoogle import Aiogoogle

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
import aiohttp

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

async def perspectiveAnalyze(comment):
	analyze_request = {
	  'comment': { 'text': comment},
	  'requestedAttributes': {'TOXICITY': {}, 'IDENTITY_ATTACK': {}, 'INSULT': {}, 'PROFANITY': {}, 'THREAT': {}},
	  'languages': ["en"]
	}
	async with Aiogoogle(api_key=secrets["perspectiveApiKey"],) as aiogoogle:
		perspectiveClient = await aiogoogle.discover('commentanalyzer', 'v1alpha1')
		response = await aiogoogle.as_api_key(
			perspectiveClient.comments.analyze(json=analyze_request)
		)
	# response = perspectiveClient.comments().analyze(body=analyze_request).execute()
	out = {}
	for trait in ["TOXICITY", "IDENTITY_ATTACK", "INSULT", "THREAT"]:
		out[trait] = response["attributeScores"][trait]["summaryScore"]["value"]
	return out
# await perspectiveAnalyze("What a waste of time. You could be making other stuff and you make this...")
# await perspectiveAnalyze("lol what the fuckk that's crazy, love it") # 0.88 for insult even though not an insult.

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


async def getCompletionOAI(*args, **kwargs):
	defaultArgs = dict(
	#   engine="text-davinci-002",
	  model="text-davinci-002",
	  temperature=0.7,
	  max_tokens=60,
	  top_p=1,
	  frequency_penalty=0,
	  presence_penalty=0,
	  stop=["Message from"],
	)
	combinedArgs = {**defaultArgs, **kwargs}
	url = "https://api.openai.com/v1/completions"
	headers={
		"Content-Type": "application/json",
		"Authorization": f"Bearer {secrets['openaiApiKey']}",
	}
	async with aiohttp.ClientSession() as session:
		async with session.post(url, json=combinedArgs, headers=headers) as r:
			responseBody = await r.json()
	responseText = responseBody["choices"][0]["text"].strip()
			
	# response = openai.Completion.create(**combinedArgs)
	# responseText = response.choices[0].text.strip()

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
async def predictFeelingGPTv1(messages, author):
	profilePrompt = makeProfilePrompt(author)
	historyPrompt = makeHistoryPrompt(messages[-5:])

	# Make full prompt with newest message
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
	prediction = await getCompletionOAI(prompt=fullPrompt, temperature=0.4)
	prediction["responseText"] = "I feel " + prediction["responseText"]
	return prediction

# Better prompting -- prompt explicitly tries to get completion that engenders empathy
async def predictFeelingGPTv2(messages, author):
	historyPrompt = makeHistoryPrompt(messages[-5:])

	# Make full prompt with newest message
	# author = messages[-2]["author"]
	authorString = authorToName(author)
	fullPrompt = textwrap.dedent(f"""
	The following is an exchange between multiple polarized parties. The last message is from another person C, who explains why {authorString} might be feeling the way they do.
	""").strip() + "\n\n" + historyPrompt + "\n\n" + textwrap.dedent(f"""
	Message from C:
	""").strip()

	prediction = await getCompletionOAI(prompt=fullPrompt, temperature=0.7, best_of=1)
	# prediction["responseText"]
	return prediction

async def predictFeelingGPTv3(messages, author1, author2):
	# author1 = messages[0]["author"]
	# author2 = messages[-1]["author"]
	historyPrompt = makeHistoryPrompt(messages[-5:])

	author1String = authorToName(author1)
	author2String = authorToName(author2)
	fullPrompt = textwrap.dedent(f"""
	In this conversation, {author1String} is vulnerable and explains to {author2String} how they really feel and why.""").strip() + "\n\n" + historyPrompt + "\n\n" + textwrap.dedent(f"""
	Message from {author1String} detailing their feelings:""").strip() + "\n\n" + "I feel"

	prediction = await getCompletionOAI(prompt=fullPrompt, temperature=0.7)
	prediction["responseText"] = "I feel " + prediction["responseText"]
	prediction["responseText"]
	return prediction

async def predictFeelingGPTv4(messages, author1, author2):
	# author1 = messages[0]["author"]
	# author2 = messages[-1]["author"]
	historyPrompt = makeHistoryPrompt(messages[-5:])

	author1String = authorToName(author1)
	author2String = authorToName(author2)
	fullPrompt = textwrap.dedent(f"""
	In this conversation, {author1String} is vulnerable and explains to {author2String} how they really feel and why.""").strip() + "\n\n" + historyPrompt + "\n\n" + textwrap.dedent(f"""
	Message from A675 detailing how B944's message made them feel, and why:""").strip() + "\n\n" + "I feel"

	prediction = await getCompletionOAI(prompt=fullPrompt, temperature=0.7)
	prediction["responseText"] = "I feel " + prediction["responseText"]
	prediction["responseText"]
	return prediction


async def predictFeelingPSP(messages):
	# [Toxicity Detection can be Sensitive to the Conversational Context]: concat parent post context to improve MAE (fig 5)
	prompt = ""
	if len(messages) > 1:
		prompt += messages[-2]["untransformedMsg"] + "\n\n"
	prompt += messages[-1]["untransformedMsg"]

	out = await perspectiveAnalyze(prompt)
	out_nocontext = await perspectiveAnalyze(messages[-1]["untransformedMsg"])
	out_combined = {"ALL": []}

	for trait in out:
		val = max(out[trait], out_nocontext[trait])
		out_combined[trait] = val
		out_combined["ALL"].append(val)

	return out_combined


# If feeling bad, include a diffuse. Sentiment one-dimensional, next step is to have model to choose amongst say ~20 feelings/scenarios -- things that are happening in the chat--, and have reactions / actions for each of those.
# Update: this does not work well for knowing when to take action.
async def evaluateFeelingIsGood(feelingString):
	# Curie is not good enough, either wrong or won't give Y/n answer
	# feelingString = "i feel disappointed and betrayed"

	prompt = textwrap.dedent(f"""
	Text:

	{feelingString}

	Is this a good feeling:""").strip()

	prediction = await getCompletionOAI(prompt=prompt, temperature=0, max_tokens=3)
	responseText = prediction["responseText"]
	if "no" in responseText.lower():
		return False
	else:
		return True

# Repectful diffuse author of messages[-2] and messages[-1], where messages[-1] is the offending message
async def respectfulDiffuse(messages):
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

	respectfulDiffusion = await getCompletionOAI(
	  prompt=respectDiffusePrompt.split("[insert]")[0],
	  suffix=respectDiffusePrompt.split("[insert]")[1],
	  temperature=0.4,
	  best_of=1,
	  max_tokens=128,
	)
	respectfulDiffusion["responseText"] = "I'm sorry " + respectfulDiffusion["responseText"]

	return respectfulDiffusion

# Transform the last message in messages to be respectful
async def transformToRespectful(messages):
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
	prediction = await getCompletionOAI(prompt=fullPrompt)
	return prediction

def qBlock(text):
	return textwrap.indent(text, "> ", lambda line: True)

async def processMessages(msgHistory):
	author1 = msgHistory[0]["author"]
	replier1 = msgHistory[-1]["author"]
	msg = msgHistory[-1]
	msg["authorFeelingsGPTv3_1"] = {author1: await predictFeelingGPTv3(msgHistory, author1, replier1)} 
	msg["authorFeelingsGPTv3_2"] = {author1: await predictFeelingGPTv3(msgHistory, author1, replier1)} 
	msg["authorFeelingsGPTv3_3"] = {author1: await predictFeelingGPTv3(msgHistory, author1, replier1)} 

	# for msg in msgHistory:
	# 	# Predict feeling for author of current
	# 	author = msg["author"]
	# 	msg["authorFeelingsGPTv2"] = {author: await predictFeelingGPTv2(msgHistory, author)}


	# transformedMsg = await transformToRespectful(msgHistory + [msg])

	# msg["transformedMsg"] = transformedMsg

	# recentAuthors = {m["author"].id: m["author"] for m in msgHistory[-20:]}
	# authorFeelings = {a: await predictFeelingGPTv2(msgHistory, a) for a in recentAuthors.values()}
	# authorFeelingsPSP = await predictFeelingPSP(msgHistory)
	# authorFeelings = {msgH}
	# msg["authorFeelings"] = authorFeelings
	# msg["authorFeelingsPSP"] = authorFeelingsPSP

	# if np.max(msg["authorFeelingsPSP"]["ALL"]) > 0.2 and len(msgHistory) >= 2:
	# 	parentAuthor = msgHistory[-2]["author"]
	# 	parentFeelingTxt = msg["authorFeelings"][parentAuthor]["responseText"]
	# 	fullDiffuse = ""
	# 	fullDiffuse += "\n" + qBlock(f"**{parentAuthor.name}**: {parentFeelingTxt}")
	# 	diffuse = await respectfulDiffuse(msgHistory)
	# 	fullDiffuse += "\n\n*" + diffuse["responseText"] + "*"
	# 	msg["fullDiffuse"] = fullDiffuse
	# 	msg["diffuse"] = diffuse

def debugViewMsgHistory(msgHistory, keys):
	# Takes msgHistory and outputs and html css grid table of the messages
	html = """
		<style>
		.item {border: 1px solid black; padding: 10px}
		.jp-OutputArea-child {display: block; flex-direction: column}
		</style>
	"""
	html += "<div style=" + f"""'display: grid; grid-template-columns: 60px repeat({1+len(keys)}, 150px); grid-gap: 0px;'""" + ">"

	html += f"""
		<div class="item">Author</div>
		<div class="item">Message</div>
	"""
	for key in keys:
		html += """<div class="item">""" + key + "</div>"
	
	for msg in msgHistory:
		html += f"""
			<div class="item">{msg["author"].name}</div>
			<div class="item">{msg["untransformedMsg"]}</div>
		"""
		for key in keys:
			divs = []
			if key not in msg:
				divs.append("""<div class="item"></div>""")
			elif isinstance(msg[key], dict) and hasattr(list(msg[key].keys())[0], "name"):
				# Per Author
				div = "<div class='item'>"
				for author in msg[key]:
					feeling = msg[key][author]
					# when user hovers, shows prompt as tooltip
					div += f"""
						<div class='item' title='{escape(feeling["promptDebug"])}'><b>{author.name}</b>: {feeling["responseText"]}</div>
					"""
				div += "</div>"
				divs.append(div)
			elif "responseText" in msg[key]:
				# Single Completion
				divs.append(f"""
					<div class="item" title='{escape(msg[key]["promptDebug"])}'>{msg[key]["responseText"]}</div>
				""")
			else:
				divs.append(f"""
					<div class="item">{msg[key]}</div>
				""")
			for div in divs:
				html += div

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
	await processMessage(msg, msgHistoryDiscordBot)

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

async def testExchange(inlines):
	# inlines = """
	# A: so, anyone see the new marvel movie?
	# B: yeah, Endgame was terrible.
	# """
	msgHistory = []
	lines = inlines.strip().split("\n")
	for line in lines:
		authName = line[:1]
		msgTxt = line[3:]
		ctx = fakeCtx(fakeAuthor(authName))
		msg = {"untransformedMsg": msgTxt, "author": ctx.author}
		msgHistory.append(msg)
	await processMessages(msgHistory)
	debugViewMsgHistory(msgHistory, ["authorFeelingsGPTv3_1", "authorFeelingsGPTv3_2", "authorFeelingsGPTv3_3"])

#%%#

testConvos = []
testConvos.append(testExchange("""
A: so, anyone see the new marvel movie?
B: yeah, Endgame was terrible.
"""))

testConvos.append(testExchange("""
A: ... To Be Continued. Please join us in creating lore and experimenting with new tools for storytelling & collaboration! http://discord.gg/XfBPAxv
B: This project was literally a soft rug pull
"""))

testConvos.append(testExchange("""
A: My company Pipedream just raised $1.6M pre-seed from @balajis to build the future of 15 min delivery. It’s like GoPuff meets Elon's The Boring Company. And we're doing it with robots racing through 12 inch pipes underground.
B: Great name for a failed enterprise!
C: Eat my asshole @A. It’s the only hole I’ll ever let you near.
"""))

testConvos.append(testExchange("""
A: yo, how is everyone doing?
B: not much, just working on some code
A: haha, doubt it’ll run
"""))

# testConvos.append(testExchange("""
# A: hi
# B: shutup
# B: hey, sorry was just joking. say that to everyone haha
# A: oh ok no worries
# """))

testConvos.append(testExchange("""
A: There will be an NFT of this image on June 10th. I know some take a dim view of NFTs, and I share a lot of those misgivings: this use of blockchain is still early. In 5-10 years crypto will inevitably evolve a stronger foundation. This would follow the pattern of all innovations.
B: Crypto will never "evolve a stronger foundation", it will always require insane amounts of energy, burning our planet. It's a pyramid scheme, not an "innovation". Why would 5-10 years matter? They've been around for twice that long, and still of no use. Sad to see you pushing it.
C: For fuck's sake, crypto is as innovative as juicero.
D: Man that's a bummer to see you buy into the grift too. Not surprising I guess, but disappointing.
"""))

testConvos.append(testExchange("""
A: There will be an NFT of this image on June 10th. I know some take a dim view of NFTs, and I share a lot of those misgivings: this use of blockchain is still early. In 5-10 years crypto will inevitably evolve a stronger foundation. This would follow the pattern of all innovations.
B: Man that's a bummer to see you buy into the grift too. Not surprising I guess, but disappointing.
"""))

testConvos.append(testExchange("""
A: Last night one of the AI developers behind that project that was ripping off living artists’ styles sent me a bunch of DMs(mostly omitted for length). He blocked me immediately after I responded and called me a moralist because I care about artists rights lol. The image sets these AI are trained on need to be public facing and opt in only. The onus needs to be on the AI devs to ethically source the images they train them with, not on the artists to keep cutting the head off the endless AI hydra appropriating our work.
B: You keep, consistently, publishing disinformation to a significant platform about this software, how it works, and the people involved in it. It's clearly not accidental - basic fact checking is not difficult. What do you gain from this hysteria you've whipped up?
"""))

testConvos.append(testExchange("""
A: Last night one of the AI developers behind that project that was ripping off living artists’ styles sent me a bunch of DMs(mostly omitted for length). He blocked me immediately after I responded and called me a moralist because I care about artists rights lol. The image sets these AI are trained on need to be public facing and opt in only. The onus needs to be on the AI devs to ethically source the images they train them with, not on the artists to keep cutting the head off the endless AI hydra appropriating our work.
B: he didnt exactly give an explanation of what he actually *does* as far as i can tell. sure "developer" wasn't the right word, but like RJ said in that thread, there wasnt really an appropriate term. like what? "dude who feeds things into AI and see what it does"?
C: he didnt exactly give an explanation of what he actually *does* as far as i can tell. sure "developer" wasn't the right word, but like RJ said in that thread, there wasnt really an appropriate term. like what? "dude who feeds things into AI and see what it does"?
"""))

testConvos.append(testExchange("""
A: Last night one of the AI developers behind that project that was ripping off living artists’ styles sent me a bunch of DMs(mostly omitted for length). He blocked me immediately after I responded and called me a moralist because I care about artists rights lol. The image sets these AI are trained on need to be public facing and opt in only. The onus needs to be on the AI devs to ethically source the images they train them with, not on the artists to keep cutting the head off the endless AI hydra appropriating our work.
B: oh are morals bad now
C: Morals are always getting in the way of my desired capitalist hellscape
"""))

testConvos.append(testExchange("""
A: I am asking people on AI discords why AI is needed. Interestingly, one response seems to be jealously or even anger that artists earned a skill others don't have
B: Man this attitude sucks so much. Part of being creative is going through the struggle of actually creating. If you microwave a meal, you arent a cook.
C: Microwaving a meal doesn’t make you a cook, but it still makes food. Why should it matter how much struggle is involved in creating something? Would you intentionally not use the undo button so that all your brushstrokes are permanent?
D: Its not about struggle its about intent. A true AI would have that intent, it would BE a person with the capability to actually use the data it has within it thoughtfully, but this isn't that. This isn't expression, it's regurgitation, and most importantly it's built on theft.
"""))

testConvos.append(testExchange("""
A: People should be able to give themselves eye exams at home and then buy prescription eyeglasses on the internet from generic suppliers.
B: I don’t want to share the road with people who need vision correction but haven’t been given an eye exam by a professional.
"""))

testConvos.append(testExchange("""
A: The case for intellectual creation in the making of an AI image using MidJourney. Part of the work is in the prompt, it takes time to know what to ask. Here's the prompt "futuristic city under a dome digital art deviantart high detail high definition octane render"
B: A hard day at work. One second: [google link to search of futuristic city under a dome]. And with this I have finished my work for today, I am going to sleep.
"""))

# This thread from Simon Stalenhag really hurts me (and I'm sure others too, in the same visceral way). How can it be diffused -- is the only way to block / hide the thread? I am not enlightened.
testConvos.append(testExchange("""
A: If you wonder why I'm not as active on here anymore, below is what my social media experience is mostly these days (and I do use the mute feature). Thanks and love to all of you who support me but I can't help it - I truly hate the world we now live in with every cell of my body.
A: And while I'm here: I really hesitate to say anything about the legal implications of AI art but let me just say I have always felt (and stated so here in the past) that taking ideas from other artists is a cornerstone of a living, artistic culture. It's the impetus for new art.
A: AI-art is really a new type of problem. I personally don't feel overly threatened by what it produces, because it is so obviously derivative. But I no longer rely on freelance work. And back when I did, I hated it specifically beause most clients, long before ai was a thing.
A: What I don't like about ai-tech is not that it can produce brand new 70s rock hits like "Keep On Glowing, You Mad Jewel" by Fink Ployd, but how it reveals that that kind of derivative, generated goo is what or new tech lords are hoping to feed us in their vision of the future.
A: Anyway, I think ai-art, just like nfts, is a technology that just amplify all the shit I hate with being an artist in this feudal capitalist dystopia where every promising new tool always end up in the hands of the least imaginative and most exploitative and unscrupulous people.
"""))

await asyncio.gather(*testConvos)

# %%
