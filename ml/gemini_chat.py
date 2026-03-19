from google import genai

client = genai.Client(api_key="AIzaSyAZxNwkAo_dNldwf0aRzecHmabOOg4J8TI")

def ask_ai(prompt):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"
