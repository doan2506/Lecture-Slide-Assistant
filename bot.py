import discord
import os
import aiohttp
from dotenv import load_dotenv
from discord.ext import commands
from discord import app_commands

load_dotenv()
token = os.environ.get("DISCORD_BOT_TOKEN")

intents = discord.Intents.default()
intents.message_content = True

class Bot(commands.Bot):
    async def setup_hook(self):
        await self.tree.sync()

bot = Bot(command_prefix='!', intents=intents)
@bot.tree.command(name='ask', description='Ask a question')
async def ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer()
    API_URL = "http://127.0.0.1:8000/ask"
    payload = {"question": question}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(API_URL, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    answer = data.get("answer", "An error happened")
                    await interaction.followup.send(f"{answer}")
                else:
                    await interaction.followup.send(f"Error: {response.status}")          
    except Exception as e:
        print(f"Error when calling API: {e}")
        await interaction.followup.send("An error happened")

bot.run(token)