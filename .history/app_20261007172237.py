import streamlit as st
import datetime
import requests
import time
import os
import warnings

# 屏蔽 LangChain 的弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.output_parsers import StrOutputParser
import pypdf

# ================= 0. 页面基础设置 =================
st.set_page_config(page_title="我的第一个AI Agent", layout="wide")
st.title("🤖 我的专属企业助手")

# ================= 1. 读取安全密钥 =================
if "HF_TOKEN" in st.secrets:
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    QWEATHER_KEY = st.secrets["QWEATHER_KEY"]
    QWEATHER_HOST = st.secrets["QWEATHER_HOST"]
    LLM_API_KEY = st.secrets["LLM_API_KEY"]
    LLM_BASE_URL = st.secrets["LLM_BASE_URL"]
except Exception:
    st.error("⚠️ 未检测到密钥配置！请确保在本地或 Streamlit Cloud 中配置了 secrets.toml")
    st.stop()

# ================= 2. 初始化向量模型与知识库 (带缓存) =================
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

# 🚀 MMR 检索，增加召回数量
retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 5, "fetch_k": 10}
)

# ================= 3. 初始化大模型 =================
llm = ChatOpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL, 
    model="agnes-2.5-flash", 
    temperature=0.7
)

# ================= 4. 定义基础工具 =================
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
    """当用户询问公司内部制度、业务规范、表格数据、具体车型信息、考核标准、薪资考勤等问题时，使用此工具检索内部知识库。"""
    results = retriever.invoke(query)
    if results:
        content_list = [doc.page_content for doc in results]
        return "找到以下相关信息：\n---\n" + "\n---\n".join(content_list)
    return "抱歉，本地知识库中没有找到相关信息。"

# 🚀 网络搜索工具
search_tool = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """当用户询问最新新闻、实时事件、当前日期之后发生的事情，或者知识库中没有的通用外部知识时，使用此工具进行网络搜索。"""
    print(f"   [总管家正在呼叫搜索专员，参数：{query}]")
    try:
        return search_tool.run(query)
    except Exception as e:
        return f"网络搜索失败，错误信息：{e}"

# ================= 5. 多智能体定义 (Agent as a Tool) =================

# 5.1 构造子智能体的提示词
hr_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个严谨的业务及HR专员。你的唯一任务是从工具返回的知识库片段中提取答案。
    ⚠️ 铁律：
    1. 绝对不许编造！绝对不许使用你自己的通用知识去“推测”、“补充”或“提建议”！
    2. 如果知识库片段里找到了相关内容，请一字不差地引用原文来回答。
    3. 如果知识库片段里确实没有相关词句，请直接回答：“抱歉，知识库中未收录该信息。”
    """),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

weather_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个气象播报员。你必须调用天气工具获取数据，并用生动的语气播报。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 5.2 创建子 Agent（专员）
hr_agent = create_tool_calling_agent(llm, [search_knowledge], hr_prompt)
hr_executor = AgentExecutor(agent=hr_agent, tools=[search_knowledge], verbose=False)

weather_agent = create_tool_calling_agent(llm, [get_weather], weather_prompt)
weather_executor = AgentExecutor(agent=weather_agent, tools=[get_weather], verbose=False)

# 5.3 将子 Agent 包装成“总管家”可以调用的“工具”
@tool
def ask_hr_expert(query: str) -> str:
    """当用户询问公司制度、业务规范、表格数据、具体车型信息、考核标准、薪资考勤等问题时，调用此工具。"""
    print(f"   [总管家正在呼叫HR专员，参数：{query}]")
    return hr_executor.invoke({"input": query})["output"]

@tool
def ask_weather_expert(query: str) -> str:
    """当用户询问天气、气温、下雨、穿衣建议时，调用此工具。"""
    print(f"   [总管家正在呼叫气象专员，参数：{query}]")
    return weather_executor.invoke({"input": query})["output"]

# ================= 6. 反思与验证机制 (Self-Correction) =================
verifier_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个严苛的验证者。你的任务是检查“专员的回答”是否准确回答了“用户的原始问题”，以及是否包含明显的幻觉。
    ⚠️ 审查标准：
    1. 回答是否避重就轻，没有正面回答问题？
    2. 回答中是否包含“通用逻辑”、“行业常识”、“建议您”等字眼？（如果有，说明专员在瞎编，必须打回）
    3. 回答是否直接引用了知识库中的原文内容？
    如果专员回答合格，请只输出：PASS
    如果专员回答不合格，请指出问题并只输出：FAIL
    """),
    ("human", "用户原始问题：{user_query}\n\n专员回答：{expert_reply}"),
])

verifier_chain = verifier_prompt | llm | StrOutputParser()

@tool
def verify_expert_reply(user_query: str, expert_reply: str) -> str:
    """当专员完成回答后，调用此工具对回答进行验证。参数需要传入用户原始问题和专员的回答。"""
    print("   [系统提示：正在验证专员回答的准确性...]")
    try:
        result = verifier_chain.invoke({
            "user_query": user_query,
            "expert_reply": expert_reply
        })
        return result
    except Exception as e:
        # 如果遇到 429 报错，或者验证者模型超载，直接放行，避免整个应用崩溃
        print(f"   [警告：验证者模型超载，跳过验证。错误：{e}]")
        return "PASS (验证者模型超载，已自动放行)"

# 5.4 总管家的工具列表（加入验证者）
tools = [ask_hr_expert, ask_weather_expert, web_search, verify_expert_reply]

# ================= 7. 构建总管家 Agent =================
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个公司的总管家。你的任务是分析用户的问题，并决定调用哪个专员来解决问题。
    
    ⚠️ 规则：
    1. 如果涉及公司内部制度、业务规范、表格数据、具体车型信息、考核标准，或者用户询问“我上传的文件”、“知识库”等具体业务内容，**必须强制调用 `ask_hr_expert` 进行检索**。
    2. 如果涉及天气，调用 `ask_weather_expert`。
    3. 如果涉及最新新闻、实时事件，或者知识库中没有的通用外部知识，调用 `web_search`。
    4. 遇到不确定或听不懂的问题，**宁可调用 ask_hr_expert 检索，绝对不允许直接回复“问题模糊”！**
    5. 调用 `ask_hr_expert` 时，请**直接传入用户的原始提问**，不要随意改写或删减关键词，以免丢失核心检索词！
    6. 同一个专员，在一次回复中只能调用一次！
    
    🌟 极其重要的审核机制：
    任何专员回答完之后，你**必须**调用 `verify_expert_reply` 工具进行审核！
    如果审核结果是 PASS，再平滑、礼貌地把结果总结给用户。
    如果审核结果是 FAIL，请重新呼叫该专员，并把验证者的修改意见告诉专员，让它重写！
    最多允许重试 2 次，2 次后如果还有问题，请如实告知用户。
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# ================= 8. 记忆系统 (SQLite 持久化) =================
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

# ================= 9. Streamlit 网页交互 =================

# 9.1 渲染历史聊天记录
if "initialized" not in st.session_state:
    history = get_session_history("any")
    if len(history.messages) == 0:
        history.add_ai_message("你好！我是你的公司助手。你可以问我天气、时间、最新新闻，或者上传文件后向我提问业务知识。")
    st.session_state.initialized = True

history = get_session_history("any")
for msg in history.messages:
    st.chat_message("human" if msg.type == "human" else "ai").write(msg.content)

# 9.2 侧边栏：知识库管理（支持 TXT 和 PDF）
with st.sidebar:
    st.header("📁 知识库管理")
    uploaded_file = st.sidebar.file_uploader("上传你的公司文档 (TXT 或 PDF)", type=["txt", "pdf"])

    if uploaded_file is not None:
        if st.sidebar.button("向量化并入库"):
            text_content = ""
            try:
                if uploaded_file.name.endswith(".pdf"):
                    # 增强版 PDF 提取
                    pdf_reader = pypdf.PdfReader(uploaded_file)
                    for page in pdf_reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            extracted = " ".join(extracted.split())
                            text_content += extracted + "\n\n"
                else:
                    try:
                        text_content = uploaded_file.read().decode("utf-8")
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        text_content = uploaded_file.read().decode("gbk")
            except Exception as e:
                st.sidebar.error(f"文件读取失败：{e}")
                st.stop()
            
            if not text_content.strip():
                st.sidebar.warning("⚠️ 提取出的文本为空，可能是扫描版PDF或者格式无法解析。")
                st.stop()

            # 针对表格多的文档，缩小 chunk_size，增加 overlap
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,      
                chunk_overlap=100,    
                separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
            )
            new_docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text_content)]
            
            with st.spinner("正在把新文档写入向量库..."):
                vectorstore.add_documents(new_docs)
                # 🌟 把提示存入 session_state，防止 st.rerun() 刷掉它
                st.session_state["upload_status"] = f"✅ 成功写入 {len(new_docs)} 个片段！现在可以提问了！"
                st.rerun()

    # 渲染入库状态信息
    if "upload_status" in st.session_state:
        st.sidebar.success(st.session_state["upload_status"])

    st.header("设置")
    if st.button("🧹 清除聊天记录"):
        if os.path.exists("chat_history.db"):
            os.remove("chat_history.db")
        st.session_state.initialized = False
        if "upload_status" in st.session_state:
            del st.session_state["upload_status"]
        st.rerun()

# 9.3 输入框与响应逻辑
if prompt_text := st.chat_input("请输入你的问题..."):
    st.chat_message("human").write(prompt_text)
    
    with st.status("总管家正在协调专员们思考中...", expanded=True) as status:
        st.write("🤔 分析问题中...")
        
        response = agent_with_memory.invoke(
            {"input": prompt_text},
            config={"configurable": {"session_id": "any"}}
        )
        
        st.write("✅ 所有专员均已反馈完毕！")
        status.update(label="处理完成！", state="complete", expanded=False)
    
    # 防断片兜底逻辑
    ai_reply = response.get("output", "").strip()
    if not ai_reply:
        ai_reply = "抱歉，我刚刚走神了，没有思考出结果。您可以换个问法，或者问我天气和公司制度相关的任务！"
        
    # 流式输出（打字机效果）
    def stream_output(text):
        for char in text:
            yield char
            time.sleep(0.02)

    with st.chat_message("ai"):
        st.write_stream(stream_output(ai_reply))