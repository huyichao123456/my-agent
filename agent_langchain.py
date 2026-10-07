import datetime
import jieba
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

# ================= 1. 初始化大模型（大脑） =================
# 建议换成你的 DeepSeek Key，如果坚持用 agnes，记得改 base_url
llm = ChatOpenAI(
    api_key="sk-R8C9A4z26gEMGokrqw1wgqkAOTdIyZQ1IPi51U0RW3qkyTKu",
    base_url="https://api.agnes-ai.cn/v1", 
    model="agnes-2.5-flash",
    temperature=0.7
)

# ================= 2. 定义本地知识库与工具（手脚） =================
knowledge_base = [
    "新员工入职满一年享有5天年假。",
    "公司每个月15号发工资。",
    "出差住宿标准是每晚不超过500元。",
    "年假可以分多次休，但每次不能少于半天。"
]

# 使用 @tool 装饰器，函数的文档字符串（docstring）就是给大模型看的说明书！
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。参数 city 是城市名称，例如：广州、北京。"""
    if city == "广州":
        return "广州今天晴，气温 25-30度，适合出门。"
    elif city == "北京":
        return "北京今天多云，气温 15-20度，有点凉。"
    return f"抱歉，暂时没有{city}的天气数据。"

@tool
def get_current_time() -> str:
    """获取当前的具体日期和时间。不需要参数。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def search_knowledge(query: str) -> str:
    """当用户询问公司内部制度、考勤、薪资、休假等问题时，使用此工具检索内部知识库。参数 query 是检索关键词。"""
    keywords = list(jieba.cut(query))
    scored_results = []
    for text in knowledge_base:
        score = sum(1 for kw in keywords if len(kw.strip()) > 0 and kw in text)
        if score > 0:
            scored_results.append((score, text))
    scored_results.sort(key=lambda x: x[0], reverse=True)
    final_texts = [item[1] for item in scored_results]
    if final_texts:
        return "找到以下相关信息：\n" + "\n".join(final_texts)
    return "抱歉，本地知识库中没有找到相关信息。"

# 把所有工具装进一个列表
tools = [get_weather, get_current_time, search_knowledge]

# ================= 3. 构建提示词（灵魂） =================
# 这里的 {agent_scratchpad} 是 LangChain 的魔法占位符，专门用来装中间步骤（工具调用、工具结果）的
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的AI助手。如果用户问天气或时间，必须调用工具查询。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ================= 4. 创建 Agent 与执行器（自动挡） =================
# 将大脑、手脚、灵魂绑在一起
agent = create_tool_calling_agent(llm, tools, prompt)
# AgentExecutor 会自动帮我们处理那层繁琐的内层 while 循环！
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ================= 5. 主循环 =================
print("开始聊天吧！(输入 '退出' 结束对话)")
while True:
    user_input = input("\n你: ")
    if user_input == "退出":
        break
    
    # 传入用户输入，AgentExecutor 自动处理所有逻辑（包括多轮工具调用）
    response = agent_executor.invoke({"input": user_input})
    print(f"\nAI: {response['output']}")