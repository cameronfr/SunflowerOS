import discord
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

class DictToObject(object):
	def __init__(self, d):
		for a, b in d.items():
			if isinstance(b, (list, tuple)):
			   setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
			else:
			   setattr(self, a, obj(b) if isinstance(b, dict) else b)

openai.api_key ="sk-L10VQgEma7q8OvDDFy87HldRb5zNfid8nvW8bEiH"

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Bot(intents=intents)

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

def getCompletionOAI(*args, **kwargs):
	defaultArgs = dict(
	  engine="text-davinci-002",
	  temperature=0.7,
	  max_tokens=30,
	  top_p=1,
	  frequency_penalty=0,
	  presence_penalty=0,
	  stop=["Message from"],
	)
	combinedArgs = {**defaultArgs, **kwargs}
	print(combinedArgs)
	response = openai.Completion.create(**combinedArgs)
	responseText = response.choices[0].text.strip()
	return responseText

# Predict feeling of author:
def predictFeeling(messages, author):
	#How this message makes barney129 feel, and why:
	#How this message makes barney129 feel, and why [usually one word]
	# Message from barney129, detailing their feelings:
	# Message from barney129, detailing how the message made them feel: [makes output more specific to last message]
	# Explanation of barney129's feelings, and why:
	authorToName = lambda a: a.name + str(a.id)[:3]
	historyPrompt = makeHistoryPrompt(messages[-5:])

	# Make full prompt with newest message
	msg = messages[-1]
	authorString = authorToName(author)
	if messages[-1]["author"].id != author.id:
		# Predict how latest message will affect others
		fullPrompt = historyPrompt + textwrap.dedent(f"""\n
		Message from {authorString} detailing how the message made them feel:
		""")
	else:
		# If latest message is author, predict how writer of that message is feeling
		fullPrompt = historyPrompt + textwrap.dedent(f"""\n
		Message from {authorString} detailing their feelings:
		""")

	print("---------- Full Prompt (Feeling) ----------\n", fullPrompt, "---------- ----------- ----------")

	# Feed into gpt-3
	responseText = getCompletionOAI(prompt=fullPrompt, temperature=0.4)
	return responseText

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

	print("---------- Full Prompt Respec ----------\n", fullPrompt, "---------- ----------- ----------")

	# Feed into gpt-3
	responseText = getCompletionOAI(prompt=fullPrompt)
	return responseText


@bot.slash_command()
async def s(ctx, msg: str):
	await ctx.defer()
	# author = {"name": ctx.author.name, "id": ctx.author.id}
	msg = {"untransformedMsg": msg, "author": ctx.author}
	# msg = {"untransformedMsg": msg, "author": "none"}
	transformedMsg = transformToRespectful(msgHistory + [msg])

	msg["transformedMsg"] = transformedMsg
	msgHistory.append(msg)

	recentAuthors = {m["author"].id: m["author"] for m in msgHistory[-20:]}
	authorFeelings = {a: predictFeeling(msgHistory, a) for a in recentAuthors.values()}
	msg["authorFeelings"] = authorFeelings

	debugChannel = bot.get_channel(1003782764484632596)

	# Format stuff
	feelingsString = ""
	for a in authorFeelings:
		feelingIsGood = "Good" if evaluateFeelingIsGood(authorFeelings[a]) else "Bad"
		feelingsString += textwrap.indent(f"\n**f_{a.name}**: {authorFeelings[a]}, **{feelingIsGood}**", "\t")
	indentedBlock = (f"""
	**Transf**: {msg['transformedMsg'].strip()}
	{feelingsString}
	""").strip()
	printOut = f"""**{ctx.author.name}**: {msg['untransformedMsg']}
	{textwrap.indent(indentedBlock, '    ')}"""

	await debugChannel.send(printOut)
	await ctx.respond(printOut)

# Dev command to simulate a message from another user
@bot.slash_command()
async def sim(ctx, auth: str, msg: str):
	author = DictToObject({"name": auth[:-3], "id": auth[-3:]})
	ctx.author = author
	await s(ctx, msg)

# Dev command to rerun last prediction
@bot.slash_command()
async def srerun(ctx):
	await ctx.defer()
	transformedMsg = transformToRespectful(msgHistory)
	authorName = msgHistory[-1]["author"].name
	await ctx.respond(f"**{authorName}**: {transformedMsg}\n*Orig: {msgHistory[-1]['untransformedMsg']}*")


# @bot.event
# async def on_message(message):
#     if message.author == bot.user:
#         return
#     await message.delete()
#     processMessage(message.author, message.content)




task = asyncio.get_event_loop().create_task(bot.start("MTAwMzczODAyNTYzMDYyOTkxOQ.GOTcQo.iCpIFd0UhrS-nhWHo3BzvHCCOd5O96nPpN0vTM"))
