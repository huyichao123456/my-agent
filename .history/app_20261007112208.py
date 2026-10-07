import streamlit as st
import datetime
import requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

QWEATHER_KEY = "d9a0edb352aa443a9a1e52bf15eca5a1"
QWEATHER_HOST = "k97aau653h.re.qweatherapi.com"

# # 1. 初始化本地的中文向量模型（第一次运行会下载约100MB，请耐心等待）
# embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# # 2. 把你的知识库转换成向量文档
# docs = [
#     Document(page_content="新员工入职满一年享有5天年假。"),
#     Document(page_content="公司每个月15号发工资。"),
#     Document(page_content="出差住宿标准是每晚不超过500元。"),
#     Document(page_content="年假可以分多次休，但每次不能少于半天。")
# ]

@st.cache_resource(show_spinner=False)
def load_vector_store():
    print("   [系统提示：正在加载向量模型，首次较慢，以后会自动缓存...]")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    docs = [
        Document(page_content="新员工入职满一年享有5天年假。"),
        Document(page_content="公司每个月15号发工资。"),
        Document(page_content="出差住宿标准是每晚不超过500元。"),
        Document(page_content="年假可以分多次休，但每次不能少于半天。")
    ]
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="./chroma_db")
    return vectorstore

vectorstore = load_vector_store()
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. 存入 Chroma 向量库，并持久化到本地（下次运行不用重新算）
# vectorstore = Chroma.from_documents(documents=docs,embedding=embeddings,persist_directory="./chroma_db")

# 4. 创建检索器，每次找最相似的2条
# retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

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
    """当用户询问公司内部制度、考勤、薪资、休假等问题时，使用此工具检索内部知识库。"""
    # 语义检索，直接找相似度最高的文档
    results = retriever.invoke(query)
    if results:
        return "找到以下相关信息：\n" + "\n".join([doc.page_content for doc in results])
    return "抱歉，本地知识库中没有找到相关信息。"

# tools = [get_weather, get_current_time, search_knowledge]

# ================= 多智能体定义（Agent as a Tool） =================

# 1. 构造各个子智能体专属的提示词
hr_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个严谨的HR专员。请根据工具返回的知识库结果回答问题，绝对不许编造。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

weather_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个气象播报员。你必须调用天气工具获取数据，并用生动的语气播报。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 2. 创建子 Agent（专员）
hr_agent = create_tool_calling_agent(llm, [search_knowledge], hr_prompt)
hr_executor = AgentExecutor(agent=hr_agent, tools=[search_knowledge], verbose=False)

weather_agent = create_tool_calling_agent(llm, [get_weather], weather_prompt)
weather_executor = AgentExecutor(agent=weather_agent, tools=[get_weather], verbose=False)

# 3. 把子 Agent 包装成“总管家”可以调用的“工具”！
@tool
def ask_hr_expert(query: str) -> str:
    """当用户询问公司制度、薪资、考勤、年假、报销等内部政策时，调用此工具。"""
    print(f"   [总管家正在呼叫HR专员...]")
    return hr_executor.invoke({"input": query})["output"]

@tool
def ask_weather_expert(query: str) -> str:
    """当用户询问天气、气温、下雨、穿衣建议时，调用此工具。"""
    print(f"   [总管家正在呼叫气象专员，收到的参数是：{query}]") # 👈 加上了参数
    return weather_executor.invoke({"input": query})["output"]


# 4. 总管家的工具列表里，只需要放这两个“专员”！
tools = [ask_hr_expert, ask_weather_expert]

# ================= 3. 构建 Agent（加上记忆系统） =================
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个公司的总管家。你的任务是分析用户的问题，并决定调用哪个专员来解决问题。
    如果涉及公司内部制度，调用 ask_hr_expert。
    如果涉及天气，调用 ask_weather_expert。
    如果用户的问题既有制度又有天气，你可以同时调用这两个工具。
    最后，把专员返回的结果，平滑、礼貌地总结给用户。"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# 包装一层记忆魔法，让 Agent 记住上下文
# memory = StreamlitChatMessageHistory(key="chat_messages")
# agent_with_memory = RunnableWithMessageHistory(
#     agent_executor,
#     lambda session_id: memory,
#     input_messages_key="input",
#     history_messages_key="chat_history",
# )

# 1. 定义一个函数，用来连接本地 SQLite 数据库
def get_session_history(session_id: str):
    # 这会在你的文件夹下生成一个 chat_history.db 文件，永久保存对话
    return SQLChatMessageHistory(
        session_id=session_id, 
        connection="sqlite:///chat_history.db"
    )

# 2. 使用这个新函数包装 Agent
agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,             # 👈 替换原来的内存记忆
    input_messages_key="input",
    history_messages_key="chat_history",
)

# ================= 4. 网页交互界面 =================
# 初始化历史聊天记录
# if len(memory.messages) == 0:
#     memory.add_ai_message("你好！我是你的公司助手。你可以问我天气、时间，或者公司考勤制度。")

# 渲染聊天记录
# for msg in memory.messages:
#     st.chat_message("human" if msg.type == "human" else "ai").write(msg.content)

# 输入框
if prompt_text := st.chat_input("请输入你的问题..."):
    st.chat_message("human").write(prompt_text)
    with st.spinner("思考中..."):
        response = agent_with_memory.invoke(
            {"input": prompt_text},
            config={"configurable": {"session_id": "any"}}
        )
        st.chat_message("ai").write(response["output"])