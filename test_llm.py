from openai import OpenAI

client = OpenAI(
    api_key="sk-R8C9A4z26gEMGokrqw1wgqkAOTdIyZQ1IPi51U0RW3qkyTKu",
    base_url="https://api.agnes-ai.cn/v1"
)

messages = [
    {"role":"system","content":"你是一个有用的AI助手。"}
]

print("开始聊天吧！（输入 '退出' 结束对话）")
while True:
    use_input= input("\n你：")
    if use_input == "退出":
        print("下次再见！")
        break

    messages.append({"role":"user","content":use_input})

    response = client.chat.completions.create(
        model="agnes-2.5-flash",
        # messages=[
        #     {"role": "user", "content": "你好！请用一句话介绍你自己。"}
        # ]
        messages=messages
    )

    reply = response.choices[0].message.content

    print(f"\nAI: {reply}")

    messages.append({"role": "assistant", "content": reply})