from openai import OpenAI

client = OpenAI(
    api_key="sk-R8C9A4z26gEMGokrqw1wgqkAOTdIyZQ1IPi51U0RW3qkyTKu",
    base_url="https://api.agnes-ai.cn/v1"
)

response = client.chat.completions.create(
    model="agnes-2.5-flash",
    messages=[
        {"role": "user", "content": "你好！请用一句话介绍你自己。"}
    ]
)

print("大模型回复：")
print(response.choices[0].message.content)