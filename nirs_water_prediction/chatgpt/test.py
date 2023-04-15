import openai
openai.api_key="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"
import os
env_vars = {
    'http_proxy': 'http://127.0.0.1:7890',
    'https_proxy': 'http://127.0.0.1:7890',
    'ftp': 'http://127.0.0.1:7890'
}
for key, value in env_vars.items():
    os.environ[key] = value
while True:
  messages = []
  # system_message = input()
  # message= {"role": "system", "content": system_message}
  # print("Alright! I am ready to be your friendly chatbot" + "\n" + "You can now type your messages.")
  message = input("")
  messages.append({"role":"user","content": message})

  response=openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages
  )
  reply = response["choices"][0]["message"]["content"]
  print()
  print()
  print()
  print(reply)
  print()
  print()
  print()