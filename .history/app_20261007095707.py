import streamlit as st
import datetime
import jieba
import requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

QWEATHER_KEY = "d9a0edb352aa443a9a1e52bf15eca5a1"
QWEATHER_HOST = "k97aau653h.re.qweatherapi.com"

# ================= 1. 基础设置 =================
st.set_page_config(page_title="我的第一个AI Agent", layout="wide")
st.title("🤖 我的专属企业助手")

# 模型初始化
llm = ChatOpenAI(
    api_key="sk-R8C9A4z26gEMGokrqw1wgqkAOTdIyZQ1IPi51U0RW3qkyTKu",
    base_url="https://api.agnes-ai.cn/v1", 
    model="agnes-2.5-flash",
    temperature=0.7
)

knowledge_base = [
    "新员工入职满一年享有5天年假。",
    "公司每个月15号发工资。",
    "出差住宿标准是每晚不超过500元。",
    "年假可以分多次休，但每次不能少于半天。"
]

# ================= 2. 定义工具 =================
@tool
def get_weather(city: str) -> str:
    """获取指定城市的今日真实天气信息。参数 city 是城市名称，例如：广州、北京。"""
    try:
        # 1. 先用 GeoAPI 把城市名翻译成 Location ID
        geo_url = f"https://{QWEATHER_HOST}/geo/v2/city/lookup?location={city}&key={QWEATHER_KEY}"
        geo_res = requests.get(geo_url, timeout=5).json()
        
        if geo_res.get("code") != "200" or not geo_res.get("location"):
            return f"抱歉，没有找到城市 {city} 的地理信息。"
            
        location_id = geo_res["location"][0]["id"]
        
        # 2. 用 Location ID 查实时天气
        weather_url = f"https://{QWEATHER_HOST}/v7/weather/now?location={location_id}&key={QWEATHER_KEY}"
        weather_res = requests.get(weather_url, timeout=5).json()
        
        if weather_res.get("code") == "200":
            now = weather_res["now"]
            return f"{city}当前天气：{now['text']}，气温：{now['temp']}℃，体感温度：{now['feelsLike']}℃，湿度：{now['humidity']}%。"
        else:
            return f"获取{city}天气失败，错误码：{weather_res.get('code')}。"
            
    except Exception as e:
        return f"查询天气时发生异常，请稍后再试。错误信息：{e}"

@tool
def get_current_time() -> str:
    """获取当前的具体日期和时间。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def search_knowledge(query: str) -> str:
    """当用户询问公司内部制度时，检索内部知识库。"""
    keywords = list(jieba.cut(query))
    scored_results = []
    for text in knowledge_base:
        score = sum(1 for kw in keywords if len(kw.strip()) > 0 and kw in text)
        if score > 0: scored_results.append((score, text))
    scored_results.sort(key=lambda x: x[0], reverse=True)
    if scored_results: return "找到以下相关信息：\n" + "\n".join([i[1] for i in scored_results])
    return "抱歉，本地知识库中没有找到相关信息。"

tools = [get_weather, get_current_time, search_knowledge]

# ================= 3. 构建 Agent（加上记忆系统） =================
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个严谨的公司AI助手。请严格遵守以下规则：
    1. 如果用户问天气或时间，必须调用相应的工具查询。天气工具只能查“今天”，如果用户问明天或未来，直接回答“抱歉，我暂时无法预测未来的天气”。
    2. 如果用户询问公司内部制度、考勤、薪资、休假、设备等问题，必须调用 search_knowledge 工具。
    3. 绝对禁止编造事实！如果你调用了 search_knowledge，并且它返回了“抱歉，本地知识库中没有找到相关信息”，你必须如实回答“抱歉，知识库中没有相关规定，建议咨询人事部门”。绝不允许使用你自己原本的知识来回答公司内部问题。
    4. 你的回答必须严格基于工具返回的结果。"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# 包装一层记忆魔法，让 Agent 记住上下文
memory = StreamlitChatMessageHistory(key="chat_messages")
agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# ================= 4. 网页交互界面 =================
# 初始化历史聊天记录
if len(memory.messages) == 0:
    memory.add_ai_message("你好！我是你的公司助手。你可以问我天气、时间，或者公司考勤制度。")

# 渲染聊天记录
for msg in memory.messages:
    st.chat_message("human" if msg.type == "human" else "ai").write(msg.content)

# 输入框
if prompt_text := st.chat_input("请输入你的问题..."):
    st.chat_message("human").write(prompt_text)
    with st.spinner("思考中..."):
        response = agent_with_memory.invoke(
            {"input": prompt_text},
            config={"configurable": {"session_id": "any"}}
        )
        st.chat_message("ai").write(response["output"])