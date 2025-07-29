import streamlit as st
import pandas as pd
import re
from collections import defaultdict

# 假设这些是你自己的模块，并且在同一目录下
# 如果这些模块不存在，你需要创建它们或注释掉相关代码
from ChatBattery.LLM_agent import LLM_Agent
from ChatBattery.domain_agent import Domain_Agent
from ChatBattery.search_agent import Search_Agent
from ChatBattery.decision_agent import Decision_Agent
from ChatBattery.retrieval_agent import Retrieval_Agent

# --- 初始设置和状态管理 ---

# 定义颜色常量
DEFAULT_COLOR = "black"
HUMAN_AGENT_COLOR = "#9A8EAF"
LLM_AGENT_COLOR = "#AC7572"
DOMAIN_AGENT_COLOR = "#DAB989"
SEARCH_AGENT_COLOR = "#8BA297"
DECISION_AGENT_COLOR = "#788BAA"
RETRIEVAL_AGENT_COLOR = "#B5C5DE"

# 初始化会话状态 (Session State)
# Streamlit 在每次交互后会重新运行脚本，st.session_state 用于在多次运行之间保存变量
def initialize_state():
    """初始化所有需要的 session_state 变量。"""
    st.session_state.setdefault('initialized', True)
    st.session_state.setdefault('conversation_list', [])
    st.session_state.setdefault('LLM_messages', [])
    st.session_state.setdefault('condition_list', [])
    st.session_state.setdefault('input_battery_list', [])
    st.session_state.setdefault('generated_text_list', [])
    st.session_state.setdefault('generated_battery_list', [])
    st.session_state.setdefault('optimal_generated_battery_list', [])
    st.session_state.setdefault('battery_record', defaultdict(str))
    st.session_state.setdefault('retrieved_battery_record', defaultdict(str))
    st.session_state.setdefault('already_started', False)
    st.session_state.setdefault('text_area_content', "请点击 '开始新会话' 按钮来启动。")

# 首次运行时初始化
if 'initialized' not in st.session_state:
    initialize_state()

# --- 辅助函数 ---

def show_content(content, color=DEFAULT_COLOR):
    """根据内容前缀设置颜色并添加到对话列表中。"""
    if content.startswith("[Human Agent]"):
        color = HUMAN_AGENT_COLOR
    elif content.startswith("[LLM Agent]"):
        color = LLM_AGENT_COLOR
    elif content.startswith("[Domain Agent]"):
        color = DOMAIN_AGENT_COLOR
    elif content.startswith("[Search Agent]"):
        color = SEARCH_AGENT_COLOR
    elif content.startswith("[Decision Agent]"):
        color = DECISION_AGENT_COLOR
    elif content.startswith("[Retrieval Agent]"):
        color = RETRIEVAL_AGENT_COLOR
    
    # 将换行符转换为 HTML 的 <br> 以便正确显示
    st.session_state.conversation_list.append({"color": color, "text": content.replace("\n", "<br>")})

@st.cache_data
def load_retrieval_DB():
    """
    加载钠离子电池数据库。
    @st.cache_data 装饰器会缓存数据，避免每次交互都重新加载。
    """
    try:
        DBfile = 'data/preprocessed.csv'
        DB = pd.read_csv(DBfile)
        DB = DB[['formula']]
        return DB
    except FileNotFoundError:
        st.error(f"错误：数据库文件未找到于 '{DBfile}'。请确保文件路径正确。")
        return pd.DataFrame({'formula': []})

def problem_conceptualization(input_battery, condition):
    """根据当前条件生成任务提示（Prompt）。"""
    mode = condition[0]

    if mode == "initial":
        task_index_prompt_template = "We have a Na cathode material FORMULA_PLACEHOLDER. Can you optimize it to develop new cathode materials with higher capacity and improved stability? You can introduce new elements from the following groups: carbon group, alkaline earth metals group, and transition elements, excluding radioactive elements; and incorporate new elements directly into the chemical formula, rather than listing them separately; and give the ratio of each element; and adjust the ratio of existing elements. My requirements are proposing five optimized battery formulations, listing them in bullet points (in asterisk *, not - or number or any other symbol), ensuring each formula is chemically valid and realistic for battery applications, and providing reasoning for each modification."
        prompt = task_index_prompt_template.replace('FORMULA_PLACEHOLDER', input_battery)

    elif mode == "update_with_generated_battery_list":
        generated_battery_list = condition[1]
        prompt = "You generated some existing or invalid battery compositions that need to be replaced with valid ones (one for each).\n"
        
        not_novel_list, invalid_list, valid_list = [], [], []
        for gen_battery in generated_battery_list:
            if st.session_state.battery_record[gen_battery] == "not novel":
                not_novel_list.append(gen_battery)
            elif st.session_state.battery_record[gen_battery] == "invalid":
                invalid_list.append(gen_battery)
            else:
                valid_list.append(gen_battery)

        if not_novel_list:
            prompt += "These batteries have been discovered before:\n" + "\n".join([f"* {x}" for x in not_novel_list]) + "\n\n"
        
        if invalid_list:
            prompt_list = []
            for inv_battery in invalid_list:
                retrieved = st.session_state.retrieved_battery_record.get(inv_battery)
                if retrieved:
                    prompt_list.append(f"* {inv_battery} (a retrieved similar and correct battery is {retrieved})")
                else:
                    prompt_list.append(f"* {inv_battery}")
            prompt += "These invalid batteries are:\n" + "\n".join(prompt_list) + "\n\n"
            
        prompt += "When replacing the invalid or existing compositions, you can replace the newly added elements with elements of lower atomic mass; and adjust the ratio of existing elements; and introduce new elements. The new compositions must be stable and have a higher capacity. The final outputs should include newly generated valid compositions, skip the retrieved batteries, and be listed in bullet points (in asterisk *, not - or number or any other symbol)."
    else:
        raise ValueError("Mode should be in [initial, update_with_generated_battery_list].")
        
    return prompt

# --- Streamlit UI 界面 ---

st.title("🔋 ChatBattery 交互式应用")

# 加载数据
retrieval_DB = load_retrieval_DB()

# --- 侧边栏控制器 ---
st.sidebar.title("控制面板")

# LLM 类型选择
llm_type = st.sidebar.selectbox(
    "选择 LLM 模型",
    ["gpt-4.1-mini", "chatgpt_o1", "chatgpt_o3"],
    key='llm_type_selector'
)

# 步骤按钮
if st.sidebar.button("▶️ 开始新会话 (Step 0)"):
    if st.session_state.already_started:
        show_content("<br><hr><br>")
    
    # 重置所有状态
    for key in st.session_state.keys():
        del st.session_state[key]
    initialize_state()

    show_content("=====" * 10)
    show_content("[ChatBattery]\n会话已重置。请输入初始电池材料，然后点击 Step 1.1。")
    st.session_state.already_started = True
    st.session_state.condition_list.append(("initial",))
    if llm_type in ["gpt-4.1-mini"]:
         st.session_state.LLM_messages = [{"role": "system", "content": "You are an expert in the field of material and chemistry."}]
    
    st.session_state.text_area_content = "在此输入你的初始电池材料 (例如: Na3V2(PO4)3)。"
    st.rerun()

st.sidebar.markdown("---")

if st.sidebar.button("Step 1.1: 问题概念化"):
    show_content("========== Step 1. 问题概念化 ==========")
    condition = st.session_state.condition_list[-1]
    
    if condition[0] == "initial":
        input_battery = st.session_state.text_area_content.strip()
        st.session_state.input_battery_list.append(input_battery)
    else:
        input_battery = st.session_state.input_battery_list[-1]
        valid_list = [b for b in condition[1] if st.session_state.battery_record[b] == "valid"]
        if valid_list:
            content = "[ChatBattery]\n上一轮的有效电池:\n" + "\n".join([f"* {x}" for x in valid_list])
            show_content(content)
    
    prompt = problem_conceptualization(input_battery, condition)
    show_content(f"[Human Agent]\n{prompt}\n\n")
    st.session_state.text_area_content = prompt
    st.rerun()

if st.sidebar.button("Step 1.2: 确认/编辑 Prompt"):
    prompt = st.session_state.text_area_content.strip()
    show_content(f"[Human Agent] (已确认)\n{prompt}\n\n")
    st.session_state.text_area_content = prompt + "\n\nNext double-check or move to Step 2.1."
    st.rerun()

st.sidebar.markdown("---")

if st.sidebar.button("Step 2.1: 生成假设"):
    show_content("========== Step 2. 生成假设 ==========")
    prompt = st.session_state.text_area_content.replace("Next double-check or move to Step 2.1.", "").strip()
    st.session_state.LLM_messages.append({"role": "user", "content": prompt})

    previous_valid_list = []
    if st.session_state.generated_battery_list:
        last_gen_list = st.session_state.generated_battery_list[-1]
        previous_valid_list = [b for b in last_gen_list if st.session_state.battery_record[b] == "valid"]

    # 调用 LLM Agent (确保 Agent 代码可用)
    generated_text, generated_battery_list = LLM_Agent.optimize_batteries(st.session_state.LLM_messages, llm_type)
    st.session_state.LLM_messages.append({"role": "assistant", "content": generated_text})
    
    current_generated_list = generated_battery_list + previous_valid_list
    st.session_state.generated_battery_list.append(current_generated_list)
    
    show_content(f"[LLM Agent]\n{generated_text}\n\n")
    st.session_state.text_area_content = "Next move to Step 2.2."
    st.rerun()

if st.sidebar.button("Step 2.2: 提取配方"):
    gen_list = st.session_state.generated_battery_list[-1]
    content = "[ChatBattery]\n请确认以下从 LLM 回复中提取的配方是否正确。"
    text_area_content = content
    for battery in gen_list:
        content += f"\n* {battery}"
        text_area_content += f"\n* {battery}"
    show_content(content + "\n\n")
    st.session_state.text_area_content = text_area_content
    st.rerun()

if st.sidebar.button("Step 2.3: 确认配方"):
    confirmed_text = st.session_state.text_area_content
    new_gen_list = []
    
    content = "[ChatBattery] (已确认)\n从LLM回复中提取的配方:"
    for line in confirmed_text.split('\n'):
        if line.startswith('*'):
            battery = line.replace('*', '').strip()
            if battery:
                new_gen_list.append(battery)
                content += f"\n* {battery}"
    
    show_content(content + "\n\n")
    st.session_state.generated_battery_list[-1] = new_gen_list
    st.session_state.text_area_content = content.replace("[ChatBattery] (已确认)", "Confirmed:") + "\n\nNext double-check or move to Step 3.1."
    st.rerun()

st.sidebar.markdown("---")

if st.sidebar.button("Step 3.1: 验证假设可行性 (DB Search)"):
    show_content("========== Step 3. 验证假设可行性 ==========")
    content = "[Search Agent]"
    gen_list = st.session_state.generated_battery_list[-1]
    
    for battery in gen_list:
        st.session_state.battery_record[battery] = "novel"
        content += f"\n\n********** 正在搜索 {battery} **********\n"
        
        # ICSD Search
        exist_icsd = Search_Agent.ICSD_search(battery, retrieval_DB["formula"].tolist())
        if exist_icsd:
            st.session_state.battery_record[battery] = "not novel"
            content += "存在于 ICSD 数据库中\n"
        else:
            content += "不存在于 ICSD 数据库中\n"
        
        # MP Search
        exist_mp = Search_Agent.MP_search(battery)
        if exist_mp:
            st.session_state.battery_record[battery] = "not novel"
            content += "存在于 MP 数据库中"
        else:
            content += "不存在于 MP 数据库中"
            
    show_content(content + "\n\n")
    st.session_state.text_area_content = "Next move to Step 4.1."
    st.rerun()

st.sidebar.markdown("---")

if st.sidebar.button("Step 4.1: 假设测试 (Domain Knowledge)"):
    show_content("========== Step 4. 假设测试 ==========")
    input_battery = st.session_state.input_battery_list[-1]
    gen_list = st.session_state.generated_battery_list[-1]

    input_value = Domain_Agent.calculate_theoretical_capacity(input_battery)
    show_content(f"[Domain Agent] 输入电池 {input_battery} 的理论容量为 {input_value:.3f}")
    
    show_content("[Decision Agent]")
    answer_list = Decision_Agent.decide_pairs(input_battery, gen_list)
    
    for gen_battery, output_value, is_valid in answer_list:
        novelty_status = "novel" if st.session_state.battery_record[gen_battery] != "not novel" else "not novel"
        validity_status = "valid" if is_valid else "invalid"
        
        show_content(
            f"* 候选电池 {gen_battery} 是 **{novelty_status}** 且 **{validity_status}**, "
            f"<span style='color:{DOMAIN_AGENT_COLOR}'>容量为 {output_value:.3f}</span>",
            color=DECISION_AGENT_COLOR
        )
        
        if novelty_status == "novel":
            st.session_state.battery_record[gen_battery] = validity_status
            if not is_valid:
                try:
                    retrieved_battery, retrieved_capacity = Retrieval_Agent.retrieve_with_domain_feedback(retrieval_DB, input_battery, gen_battery)
                    retrieved_content = (f"[Retrieval Agent] 检索到最相似的有效电池: {retrieved_battery} "
                                         f"<span style='color:{DOMAIN_AGENT_COLOR}'>容量为 {retrieved_capacity:.3f}</span>")
                    st.session_state.retrieved_battery_record[gen_battery] = retrieved_battery
                except Exception:
                    retrieved_content = "[Retrieval Agent] 未检索到相似的有效电池。"
                show_content(retrieved_content)

    all_pass = all(st.session_state.battery_record[b] == "valid" for b in gen_list)
    
    if all_pass:
        st.session_state.text_area_content = "任务完成！所有生成的电池均有效。"
        st.balloons()
    else:
        condition = ("update_with_generated_battery_list", gen_list)
        st.session_state.condition_list.append(condition)
        st.session_state.text_area_content = "部分电池无效或已存在。请返回 Step 1.1 进行迭代优化。"
    
    st.rerun()

st.sidebar.markdown("---")

# --- 主显示区域 ---

st.header("对话历史")
# 显示对话内容
for item in st.session_state.conversation_list:
    st.markdown(f'<p style="color:{item["color"]};">{item["text"]}</p>', unsafe_allow_html=True)

st.header("输入/确认区域")
st.text_area(
    "根据提示在此处输入或确认内容",
    key='text_area_content',
    height=250
)
