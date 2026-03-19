import google.generativeai as genai
import os

genai.configure(api.key=os.getenv("AIzaSyAZxNwkAo_dNldwf0aRzecHmabOOg4J8TI"))


def ask_ai(prompt):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"
