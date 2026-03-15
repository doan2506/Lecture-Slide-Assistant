import os
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "AIzaSyDxT5_ijA1DVIsb2cwY_GK2KQWblXrkYCQ"

print("⏳ Connecting to Google Generative AI API...")

try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    response = llm.invoke("Hello, are you available? Please respond in one short sentence.")
    
    print("\n✅ SUCCESS! LangChain has successfully connected to your Google Generative AI API key.")
    print("🤖 AI response:", response.content)

except Exception as e:
    print("\n❌ FAILURE! An error occurred while connecting:")
    print(e)