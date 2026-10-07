from openai import OpenAI
import json

client = OpenAI(
    api_key="sk-R8C9A4z26gEMGokrqw1wgqkAOTdIyZQ1IPi51U0RW3qkyTKu",
    base_url="https://api.agnes-ai.cn/v1"
)

def get_weather(city):
    print(f"   [系统提示：正在调用本地函数，准备获取 {city} 的天气...]")
    # 1. 模拟获取到的原始天气数据（通常在真实场景中，这里会去调第三方天气API）
    if city == "广州":
        raw_weather = "晴，气温 25-30度"
    elif city == "北京":
        raw_weather = "多云，气温 15-20度"
    else:
        raw_weather = "未知"

    if raw_weather == "未知":
        return f"抱歉，暂时没有{city}的天气数据。"

    # 2. 🌟 核心操作：在工具内部，再次调用 AI 大模型 🌟
    print(f"   [系统提示：正在调用 AI 润色 {city} 的天气信息...]")
    
    # ⚠️ 注意：这里我们要重新发一次请求，并且给它一个独立的、干净的 messages
    # 绝对不能复用外层的 messages，否则会污染主对话的历史记录！
    tool_messages = [
        {"role": "system", "content": "你是一个幽默的气象播报员。请根据用户提供的天气原始数据，用一句简短、生动、带点幽默的话播报出来。"},
        {"role": "user", "content": f"城市：{city}，天气数据：{raw_weather}"}
    ]

    try:
        # 再次请求大模型（可以换成更便宜的模型来省钱）
        response = client.chat.completions.create(
            model="agnes-2.5-flash", # 这里可以用你之前能跑通的那个模型
            messages=tool_messages
        )
        # 获取 AI 润色后的结果
        ai_polished_text = response.choices[0].message.content
        return ai_polished_text
    except Exception as e:
        # 如果第二次调用 AI 失败了，作为兜底，直接返回原始数据
        print(f"   [警告：AI 润色失败，使用原始数据。原因：{e}]")
        return f"{city}今天是{raw_weather}。"

# 3. 给大模型看的“工具说明书”
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：广州、北京",
                    }
                },
                "required": ["city"], # 必填参数
            },
        }
    }
]

# 4. 初始化记忆列表
messages = [{"role": "system", "content": "你是一个天气助手。如果用户问天气，你必须调用工具查询。"}]

print("开始聊天吧！(输入 '退出' 结束对话)")

while True:
    user_input = input("\n你: ")
    if user_input == "退出":
        break
    
    messages.append({"role": "user", "content": user_input})
    
    # 5. 第一次请求大模型：带上工具说明书
    response = client.chat.completions.create(
        model="agnes-2.5-flash", # 注意：模型必须支持 Function Calling
        messages=messages,
        tools=tools,
        tool_choice="auto" # 让模型自己决定要不要用工具
    )
    
    response_message = response.choices[0].message
    
    # 6. 判断大模型是否决定调用工具
    if response_message.tool_calls:
        print(f"\n[大模型决定调用工具：{response_message.tool_calls[0].function.name}]")
        
        # 提取模型给我们的参数（比如 {"city": "广州"}）
        function_name = response_message.tool_calls[0].function.name
        function_args = json.loads(response_message.tool_calls[0].function.arguments)
        
        # 7. 在本地执行真正的 Python 函数
        if function_name == "get_weather":
            city = function_args.get("city")
            tool_result = get_weather(city)
            print(f"   [本地函数返回结果：{tool_result}]")
            
            # 8. 把大模型请求调用工具这一步，加入记忆
            messages.append(response_message)
            
            # 9. 把工具的执行结果，作为新消息加入记忆（role 必须是 tool）
            messages.append({
                "role": "tool",
                "tool_call_id": response_message.tool_calls[0].id,
                "content": tool_result
            })
            
            # 10. 第二次请求大模型：让它根据工具查到的真实数据，总结成人话回复你
            second_response = client.chat.completions.create(
                model="agnes-2.5-flash",
                messages=messages
            )
            final_reply = second_response.choices[0].message.content
            print(f"\nAI: {final_reply}")
            
            # 把最终的回复也存进记忆
            messages.append(second_response.choices[0].message)
            
    else:
        # 如果大模型觉得不需要查工具（比如你只是打招呼），直接回复
        print(f"\nAI: {response_message.content}")
        messages.append(response_message)