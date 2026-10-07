from openai import OpenAI
import json
import datetime

client = OpenAI(
    api_key="sk-R8C9A4z26gEMGokrqw1wgqkAOTdIyZQ1IPi51U0RW3qkyTKu",
    base_url="https://api.agnes-ai.cn/v1"
)

# 模拟企业内部的文档库
knowledge_base = [
    "新员工入职满一年享有5天年假。",
    "公司每个月15号发工资。",
    "出差住宿标准是每晚不超过500元。",
    "年假可以分多次休，但每次不能少于半天。"
]

def search_knowledge(query):
    print(f"   [系统提示：正在本地知识库中检索 '{query}'...]")
    keywords = query.split()

    if not keywords:
        keywords = [query]

    scored_results = []

    for text in knowledge_base:
        score = 0
        # 文档每包含一个关键词，得一分
        for kw in keywords:
            if kw in text:
                score += 1
        if score > 0:
            scored_results.append((score, text))

    # 按分数从高到低排序
    scored_results.sort(key=lambda x: x[0], reverse=True)

    # 提取最高分的文本
    final_texts = [item[1] for item in scored_results]

    if final_texts:
        return "找到以下相关信息：\n" + "\n".join(final_texts)
    return "抱歉，本地知识库中没有找到相关信息。"

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

def get_current_time():
    print("   [系统提示：正在获取当前时间...]")
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
    },
    {
        "type":"function",
        "function":{
            "name":"get_current_time",
            "description":"获取当前的具体的日期和时间",
            "parameters": {"type": "object", "properties": {}} # 这个工具不需要参数
        }
    },
    {
        "type":"function",
        "function":{
            "name": "search_knowledge",
            "description": "当用户询问公司内部制度、考勤、薪资、休假等问题时，使用此工具检索内部知识库。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用于检索的关键词，比如：年假、发工资、出差住宿标准",
                    }
                },
                "required": ["query"],
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

    # 开启一个“Agent内循环”
    while True:
    
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
            messages.append(response_message) # 把调用记录加入记忆
            # 遍历模型请求的每一个工具调用
            for tool_call in response_message.tool_calls:
            
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                print(f"\n[大模型决定调用工具：{function_name}，参数：{function_args}]")
                
                # 7. 在本地执行真正的 Python 函数
                if function_name == "get_weather":
                    tool_result = get_weather(function_args.get("city"))
                elif function_name == "get_current_time":
                    tool_result = get_current_time()
                elif function_name == "search_knowledge":
                    tool_result = search_knowledge(function_args.get("query"))
                else:
                    tool_result = "未知工具"
                    
                print(f"   [本地函数返回结果：{tool_result}]")
                
                # 把每一个工具的结果都存入记忆
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
                    
            # else:
            #     # 如果模型没有再调用工具，而是输出了文本，说明它认为任务完成了
            #     final_reply = response_message.content
            #     print(f"\nAI: {final_reply}")
            #     messages.append(response_message) # 把最终回复存入记忆
            #     break # 跳出内层循环，等待用户下一次输入

            else:
                # 检查模型是不是没说话
                if not response_message.content:
                    print("\n   [系统提示：模型未输出内容，正在强制生成最终回复...]")
                    # 追加一句催促指令，塞进记忆
                    messages.append({"role": "user", "content": "请根据以上查询结果，用一句简短的话总结并回复我。"})
                    # 再发一次请求（强制不带tools，逼它必须说话）
                    final_response = client.chat.completions.create(
                        model="agnes-2.5-flash",
                        messages=messages
                    )
                    final_reply = final_response.choices[0].message.content
                    print(f"\nAI: {final_reply}")
                    messages.append(final_response.choices[0].message)
                else:
                    # 如果模型正常说话了
                    print(f"\nAI: {response_message.content}")
                    messages.append(response_message)
                
                break # 跳出内层循环，等待用户下一次输入