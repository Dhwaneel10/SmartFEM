from google import genai
client = genai.Client(api_key="AIzaSyAZxNwkAo_dNldwf0aRzecHmabOOg4J8TI")
models = client.models.list()

for m in models:
    print(m.name)
  

