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
from html import unescape
import numpy as np

class DictToObject(object):
	def __init__(self, d):
		for a, b in d.items():
			if isinstance(b, (list, tuple)):
			   setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
			else:
			   setattr(self, a, obj(b) if isinstance(b, dict) else b)

openai.api_key =""

perspectiveApiKey = ""
perspectiveClient = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=perspectiveApiKey,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)
def perspectiveAnalyze(comment):
	analyze_request = {
	  'comment': { 'text': comment},
	  'requestedAttributes': {'TOXICITY': {}, 'IDENTITY_ATTACK': {}, 'INSULT': {}, 'PROFANITY': {}, 'THREAT': {}}
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
class MyClient(discord.Bot):
	async def on_ready(self):
		print(f"Logged in as {self.user} (ID: {self.user.id})")
		print("------")

	async def on_message(self, message: discord.Message):
		# Make sure we won't be replying to ourselves.
		if message.author.id == self.user.id:
			return
		# print("Got msg", message.content)
# bot = discord.Bot(intents=intents)
bot = MyClient(intents=intents)

# {transformedMsg: str, untransformedMsg: str, author: author obj}
msgHistory = []
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


	print("---------- Running Prompt ----------")
	print(kwargs["prompt"])
	if "suffix" in kwargs:
		print("[INSERT]" + kwargs["suffix"])
	print("--ARGS:", combinedArgs)
	print("--RESP:", responseText)
	print("---------- ----------- ----------")

	return responseText

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
	responseText = getCompletionOAI(prompt=fullPrompt, temperature=0.4)
	return "I feel " + responseText

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

	responseText = getCompletionOAI(prompt=prompt, temperature=0, max_tokens=3)
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

	feelingPrediction = predictFeelingGPT(messages, offendedAuthor)
	# feelingPrediction = "fuck you too"
	offendingName = authorToName(offendingAuthor)
	offendedName = authorToName(offendedAuthor)

	profilePrompt = makeProfilePrompt(offendingAuthor)

	respectDiffusePrompt = f"""
{profilePrompt}

{historyPrompt}

Message from {offendedName}:

{feelingPrediction}

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

	return "I'm sorry " + respectfulDiffusion

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
	responseText = getCompletionOAI(prompt=fullPrompt)
	return responseText

def qBlock(text):
	return textwrap.indent(text, "> ", lambda line: True)

@bot.slash_command()
async def s(ctx, msg: str):


	await ctx.defer()
	# author = {"name": ctx.author.name, "id": ctx.author.id}
	# msg = fakeMsg("a", "hello all");
	# ctx=fakeCtx(msg["author"])
	# msg = msgHistory[-1]
	msg = {"untransformedMsg": msg, "author": ctx.author}
	# msg = {"untransformedMsg": msg, "author": "none"}
	transformedMsg = transformToRespectful(msgHistory + [msg])

	msg["transformedMsg"] = transformedMsg
	msgHistory.append(msg)

	recentAuthors = {m["author"].id: m["author"] for m in msgHistory[-20:]}
	authorFeelings = {a: predictFeelingGPT(msgHistory, a) for a in recentAuthors.values()}
	authorFeelingsPSP = predictFeelingPSP(msgHistory)
	msg["authorFeelings"] = authorFeelings
	msg["authorFeelingsPSP"] = authorFeelingsPSP

	debugChannel = bot.get_channel(1003782764484632596)


	# Format stuff
	feelingsString = ""
	for a in msg["authorFeelings"]:
		feelingString = msg["authorFeelings"][a]
		feelingIsGood = "Good" if evaluateFeelingIsGood(feelingString) else "Bad"
		feelingsString += qBlock(f"**{a.name}**: {feelingString}, **{feelingIsGood}**\n")
	printOutDebug = textwrap.dedent(f"""
---------
**{ctx.author.name}**: \n{qBlock(msg['untransformedMsg'])}
**Transform**: \n{qBlock(msg['transformedMsg'].strip())}
**FeelPSP**: \n{qBlock(str(msg["authorFeelingsPSP"]))}
**FeelGPTPred**: \n{feelingsString}
	""").strip()
	# print(printOutDebug)

	printOut = f"**{ctx.author.name}**: " + msg["untransformedMsg"]
	if np.max(msg["authorFeelingsPSP"]["ALL"]) > 0.2:
		parentAuthor = msgHistory[-2]["author"]
		parentFeeling = msg["authorFeelings"][parentAuthor]
		printOut += "\n" + qBlock(f"**{parentAuthor.name}**: {parentFeeling}")
		diffuse = respectfulDiffuse(msgHistory)
		printOut += "\n*" + diffuse + "*"

	await debugChannel.send(printOutDebug)
	await ctx.respond(printOut)

# Dev command to simulate a message from another user
@bot.slash_command()
async def sim(ctx, auth: str, msg: str):
	author = fakeAuthor(auth)
	ctx.author = author
	await s(ctx, msg)

# Dev command to rerun last prediction
@bot.slash_command()
async def srerun(ctx):
	await ctx.defer()
	transformedMsg = transformToRespectful(msgHistory)
	authorName = msgHistory[-1]["author"].name
	await ctx.respond(f"**{authorName}**: {transformedMsg}\n*Orig: {msgHistory[-1]['untransformedMsg']}*")

# Dev command to clear history
@bot.slash_command()
async def clear(ctx):
	await ctx.defer()
	msgHistory.clear()
	await ctx.respond("History Cleared")

# For testing stuff
def fakeAuthor(name):
	id = hash(name) % 1000
	author = DictToObject({"name": name, "id": id})
	return author

def fakeCtx(author):
	ctx = DictToObject({"author": author})
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




task = asyncio.get_event_loop().create_task(bot.start(""))
