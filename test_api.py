import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

print("⏳ Connecting to Google Generative AI API...")

try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    response = llm.invoke("ListModels")
    
    print("\n✅ SUCCESS! LangChain has successfully connected to your Google Generative AI API key.")
    print("🤖 AI response:", response.content)

except Exception as e:
    print("\n❌ FAILURE! An error occurred while connecting:")  
    print(e)