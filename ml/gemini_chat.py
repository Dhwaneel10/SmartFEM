import google.generativeai as genai
import streamlit as st

# Correct way
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

def ask_ai(prompt):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)

        return response.text if response else "No response from AI"

    except Exception as e:
        return f"AI Error: {str(e)}"
