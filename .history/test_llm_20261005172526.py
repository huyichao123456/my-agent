from openai import OpenAI

client = OpenAI(
    api_key="sk-73d944cf4ecb467eb46707af75d56ef0",
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "你好！请用一句话介绍你自己。"}
    ]
)

print("大模型回复：")
print(response.choices[0].message.content)