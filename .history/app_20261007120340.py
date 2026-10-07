import streamlit as st
import datetime
import requests
import time
import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 尝试从 st.secrets 读取，读不到就报错
QWEATHER_KEY = st.secrets.get("QWEATHER_KEY", "本地默认Key")
QWEATHER_HOST = st.secrets.get("QWEATHER_HOST", "本地默认Host")

llm = ChatOpenAI(
    api_key=st.secrets.get("LLM_API_KEY", "本地默认Key"), 
    base_url=st.secrets.get("LLM_BASE_URL", "本地默认URL"), 
    model="agnes-2.5-flash",
    temperature=0.7
)

# ================= 1. 初始化向量模型与知识库 (带缓存) =================
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


# ================= 3. 定义基础工具 =================
@tool
def get_weather(city: str) -> str:
    """获取指定城市的今日真实天气信息。参数 city 是城市名称，例如：广州、北京。"""
    try:
        geo_url = f"https://{QWEATHER_HOST}/geo/v2/city/lookup?location={city}&key={QWEATHER_KEY}"
        geo_res = requests.get(geo_url, timeout=5).json()
        
        if geo_res.get("code") != "200" or not geo_res.get("location"):
            return f"抱歉，没有找到城市 {city} 的地理信息。"
            
        location_id = geo_res["location"][0]["id"]
        weather_url = f"https://{QWEATHER_HOST}/v7/weather/now?location={location_id}&key={QWEATHER_KEY}"
        weather_res = requests.get(weather_url, timeout=5).json()
        
        if weather_res.get("code") == "200":
            now = weather_res["now"]
            return f"{city}当前天气：{now['text']}，气温：{now['temp']}℃，体感温度：{now['feelsLike']}℃，湿度：{now['humidity']}%。"
        else:
            return f"获取{city}天气失败，错误码：{weather_res.get('code')}。"
    except Exception as e:
        return f"查询天气时发生异常：{e}"

@tool
def get_current_time() -> str:
    """获取当前的具体日期和时间。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def search_knowledge(query: str) -> str:
    """当用户询问公司内部制度、考勤、薪资、休假等问题时，使用此工具检索内部知识库。"""
    results = retriever.invoke(query)
    if results:
        return "找到以下相关信息：\n" + "\n".join([doc.page_content for doc in results])
    return "抱歉，本地知识库中没有找到相关信息。"

# ================= 4. 多智能体定义 (Agent as a Tool) =================

# 4.1 构造子智能体的提示词
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

# 4.2 创建子 Agent（专员）
hr_agent = create_tool_calling_agent(llm, [search_knowledge], hr_prompt)
hr_executor = AgentExecutor(agent=hr_agent, tools=[search_knowledge], verbose=False)

weather_agent = create_tool_calling_agent(llm, [get_weather], weather_prompt)
weather_executor = AgentExecutor(agent=weather_agent, tools=[get_weather], verbose=False)

# 4.3 将子 Agent 包装成“总管家”可以调用的“工具”
@tool
def ask_hr_expert(query: str) -> str:
    """当用户询问公司制度、薪资、考勤、年假、报销等内部政策时，调用此工具。"""
    print(f"   [总管家正在呼叫HR专员，参数：{query}]")
    return hr_executor.invoke({"input": query})["output"]

@tool
def ask_weather_expert(query: str) -> str:
    """当用户询问天气、气温、下雨、穿衣建议时，调用此工具。"""
    print(f"   [总管家正在呼叫气象专员，参数：{query}]")
    return weather_executor.invoke({"input": query})["output"]

# 4.4 总管家的工具列表
tools = [ask_hr_expert, ask_weather_expert]

# ================= 5. 构建总管家 Agent =================
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个公司的总管家。你的任务是分析用户的问题，并决定调用哪个专员来解决问题。
    如果涉及公司内部制度，调用 ask_hr_expert。
    如果涉及天气，调用 ask_weather_expert。
    如果用户的问题既有制度又有天气，你可以同时调用这两个工具。
    
    ⚠️ 极其重要的规则：同一个专员，在一次回复中只能调用一次！绝对不允许为了同样的问题重复呼叫同一个专员！
    
    最后，把专员返回的结果，平滑、礼貌地总结给用户。"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# ================= 6. 记忆系统 (SQLite 持久化) =================
def get_session_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id, 
        connection="sqlite:///chat_history.db"
    )

agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# ================= 7. Streamlit 网页界面 =================
st.set_page_config(page_title="我的第一个AI Agent", layout="wide")
st.title("🤖 我的专属企业助手")

# 初始化问候语（只在数据库为空时显示一次）
if "initialized" not in st.session_state:
    history = get_session_history("any")
    if len(history.messages) == 0:
        history.add_ai_message("你好！我是你的公司助手。你可以问我天气、时间，或者公司考勤制度。")
    st.session_state.initialized = True

# 渲染历史聊天记录
history = get_session_history("any")
for msg in history.messages:
    st.chat_message("human" if msg.type == "human" else "ai").write(msg.content)

# 输入框与响应逻辑
if prompt_text := st.chat_input("请输入你的问题..."):
    st.chat_message("human").write(prompt_text)
    
    # 使用 st.status 创建一个可折叠的“状态框”，让用户看到 Agent 的思考过程
    with st.status("总管家正在协调专员们思考中...", expanded=True) as status:
        st.write("🤔 分析问题中...")
        
        response = agent_with_memory.invoke(
            {"input": prompt_text},
            config={"configurable": {"session_id": "any"}}
        )
        
        st.write("✅ 所有专员均已反馈完毕！")
        status.update(label="处理完成！", state="complete", expanded=False)
        
def stream_output(text):
    for char in text:
        yield char
        time.sleep(0.02) # 调整这个数字可以改变打字速度，0.02秒/字

    with st.chat_message("ai"):
        st.write_stream(stream_output(response["output"]))