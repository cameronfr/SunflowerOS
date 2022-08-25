#%%#
from cgitb import text
import discord
from aiogoogle import Aiogoogle
import openai
import json
import textwrap
import numpy as np
import os
from IPython.display import display, HTML
import aiohttp

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

async def getCompletionOAI(*args, **kwargs):
    # kwargs = {"prompt": "test with space at end "}
    # kwargs = {"prompt": "testing123"}
    if kwargs["prompt"][-1] == " ":
        raise ValueError("Trailing space in prompt")

    defaultArgs = dict(
        #engine="text-davinci-002",
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

    prediction = {"promptDebug": promptDebug, "responseText": responseText, "fullArgs": fullArgs}

    return prediction

# %%

import snscrape.modules.twitter as sntwitter
scraper = sntwitter.TwitterTweetScraper(tweetId=1553200773164462080)
out = list(scraper.get_items())

async def reactToSignalAgreeUrge(a1Name, a1Bio, a1FollowerCt, a1FollowingCt, a1Message):
    prompt = textwrap.dedent(f"""
    In their second message, {a1Name} adds onto their first message, conveying to the reader that people who agree with their first message are peers who they respect and would work with, and strongly implying that they don't think they're better than anyone else.
    \t
    {a1Name}'s Profile:
    Follower Count: {a1FollowerCt}, Following Count: {a1FollowingCt}
    Bio: {a1Bio}
    \t
    Message from {a1Name}: 
    {a1Message}
    \t
    Message from {a1Name}:
    {a1Message}
    """).strip() + "\n"
    response = await getCompletionOAI(prompt=prompt, temperature=0.7)
    return response

async def reactToDoom(a1Name, a1Bio, a1FollowerCt, a1FollowingCt, a1Message):

    prompt = textwrap.dedent(f"""
    The following is a message from fchollet. His first message is pessimistic, but his second message adds onto the first and is very optimistic and inspiring.
    \t
    {a1Name}'s Profile:
    Follower Count: {a1FollowerCt}, Following Count: {a1FollowingCt}
    Bio: {a1Bio}
    \t
    Message from {a1Name}: 
    {a1Message}
    \t
    Message from {a1Name}:
    {a1Message}
    """).strip() + "\n"
    response = await getCompletionOAI(prompt=prompt, temperature=0.7)
    return response

async def classifyTweet(url):
    url = "https://twitter.com/dystopiabreaker/status/1553200773164462080" #dystopiabreaker
    # url = "https://twitter.com/fchollet/status/976933782367293440" #fchollet
    # url = "https://twitter.com/hasanthehun/status/1562554743846604800"


    tweetId = int(url.split("/")[-1])
    scraper = sntwitter.TwitterTweetScraper(tweetId=tweetId)
    tweetInfo = list(scraper.get_items())[0]
    a1Name = tweetInfo.user.username
    a1Bio = tweetInfo.user.description
    a1FollowerCt = tweetInfo.user.followersCount
    a1FollowingCt = tweetInfo.user.friendsCount
    a1Message = tweetInfo.content
    
    # tweetInfo.quotedTweet
    classificationOutput = await classifyWithPrompt(a1Name, a1Bio, a1FollowerCt, a1FollowingCt, a1Message)
    print(f"""{a1Name}: {a1Message}\n\n{classificationOutput["responseText"]}""")

    # print(classificationOutput["promptDebug"])

    classString = classificationOutput["responseText"].lower()
    if len(classString.split("\n")) == 1:
        if "doom" in classString:
            print("*Using doom action path*\n")
            thoughtGuideOutput = await reactToDoom(a1Name, a1Bio, a1FollowerCt, a1FollowingCt, a1Message)
            thoughtGuide = thoughtGuideOutput["responseText"]
            print(thoughtGuide)
        elif "superior" in classString or "must-share" in classString:
            print("*Using superior / shareurge action path*\n")
            thoughtGuideOutput = await reactToSignalAgreeUrge(a1Name, a1Bio, a1FollowerCt, a1FollowingCt, a1Message)
            thoughtGuide = thoughtGuideOutput["responseText"]
            print(thoughtGuide)

async def classifyWithPrompt(a1Name, a1Bio, a1FollowerCt, a1FollowingCt, a1Message):
    prompt = textwrap.dedent(f"""
        Describe which of the following categories the message from {a1Name} falls into for PersonC.
        \t
        -- Categories --
        Doom: the message creates a senes of doom or pessimism about the current state of affairs.
        Excitement-FOMO: the message makes PersonC excited, but also makes them feel that they should be doing something and that they are missing out.
        Must-share-too: the message makes PersonC feel that they need to tell everyone that they agree, like a student in class feeling the need to share that they know something right before the teacher teaches everyone. It makes the reader feel the need to let everyone know that they had the thought first.
        Unfair: the message makes PersonC feel that they are being wronged, that something unfair is happening.
        Agree-dunk: the message is dunking on someone, and PersonC feels a sense of accomplishment that their team is winning.
        Anger-dunk: the message is dunking on someone, and PersonC feels attacked because they side with the person being dunked on.
        Attack-anger: the message makes PersonC feel that some part of their identity is being attacked, even if it is just implied in the message.
        Superiority-Agreement: the message makes the PersonC feel both like they have to agree, and that in agreeing they are acknowledging the author's superiority -- e.g. because the author had the idea first -- without wanting to.
        Insecure-fear: the message makes PersonC worry about some aspect about themselves which they are insecure about, like their appearance.
        Bragging: the message comes off as bragging or a humble-brag, and makes the reader feel less skilled.
        Project-vulernability: the message is from PersonC, and they are sharing a project that they are working on, and they are being vulnerable because the project is not finished.
        Fortune-cookie: the message is satisfying to read and feels like it is offering wisdom, but doesn't really say anything concrete.
        \t
        -- Background Information --
        {a1Name}'s Profile:
        Follower Count: {a1FollowerCt}, Following Count: {a1FollowingCt}
        Bio: {a1Bio}
        \t
        -- Messages --
        Message from {a1Name}: 
        {a1Message}
        \t
        -- Classification --
        Which category does the message from {a1Name} fall into for PersonC? If it depends on PersonC's beliefs, list the possible scenarios in the form [if belief]: [category].
    """).strip()
    response = await getCompletionOAI(prompt=prompt, temperature=0.7)
    return response