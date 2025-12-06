from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict
from langchain_google_genai import GoogleGenerativeAI




load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')
# gemini-2.5-flash
# Gemini 2.5 Pro

class review(TypedDict):
    summary:str
    sentiment:str
    
with_structured_output=model.with_structured_output(review)

result=with_structured_output.invoke("""I feel like nothing is going right today, and everything around me seems to be falling apart. The weather is gloomy, and it’s affecting my mood more than I expected. I worked so hard, yet the results turned out to be incredibly disappointing. Every task I try to finish feels unnecessarily difficult, and I don’t feel motivated to continue anymore. No matter how much effort I put in, failure keeps finding its way back to me. This entire project is draining me mentally and emotionally, leaving me exhausted. I feel ignored, as if my voice and presence don’t matter to anyone. My consistent efforts seem to remain unnoticed, which makes everything feel pointless. Today feels like a complete waste of time, and I regret even getting out of bed. I’m losing hope that things will ever improve, and that scares me. I don’t understand why life insists on being so complicated. I feel stuck, unable to make any meaningful progress. My plans never work out the way I imagine them. The more I think about everything, the more frustrated and overwhelmed I become. I regret trusting people who didn’t deserve it. Nothing about this situation feels fair, and I’m tired of pretending I’m okay. I can’t shake off this persistent feeling of disappointment. My confidence seems to be decreasing with every passing day. I constantly feel like I’m letting myself down. It hurts to watch others succeed while I’m still struggling. Sometimes I wish I had never taken this path at all. The outcomes are always worse than anything I prepare for. I feel like I’m losing control of everything around me. My ideas get rejected before they are even understood. I can’t find a single reason to stay positive right now. I feel like I don’t belong anywhere, and that emptiness terrifies me. Every decision I make seems to be the wrong one. I’m exhausted from fighting the same battles repeatedly, with no hope of victory in sight""")

print(result)



