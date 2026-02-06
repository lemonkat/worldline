import os

import dotenv
import dspy

dotenv.load_dotenv()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

LM_FLASH = dspy.LM("gemini/gemini-3-flash-preview", api_key=GEMINI_API_KEY)
LM_PRO = dspy.LM("gemini/gemini-3-pro-preview", api_key=GEMINI_API_KEY)
