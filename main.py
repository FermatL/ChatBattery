import json
import argparse
import sys
import re
import pandas as pd
from collections import defaultdict
import streamlit as st

# Assuming your agent classes are in these files.
# Make sure these files are in the same directory or accessible in the Python path.
from ChatBattery.LLM_agent import LLM_Agent
from ChatBattery.domain_agent import Domain_Agent
from ChatBattery.search_agent import Search_Agent
from ChatBattery.decision_agent import Decision_Agent
from ChatBattery.retrieval_agent import Retrieval_Agent

# --- Configuration and Constants ---
# Define colors for different agents
DEFAULT_COLOR         = "black"
HUMAN_AGENT_COLOR     = "#9A8EAF"
LLM_AGENT_COLOR       = "#AC7572"
DOMAIN_AGENT_COLOR    = "#DAB989"
SEARCH_AGENT_COLOR    = "#8BA297"
DECISION_AGENT_COLOR  = "#788BAA"
RETRIEVAL_AGENT_COLOR = "#B5C5DE"

# --- Helper Functions ---

# Use Streamlit's caching to load data only once
@st.cache_data
def load_retrieval_DB():
    """Loads the retrieval database from a CSV file."""
    try:
        DBfile = 'data/Na_battery/preprocessed.csv'
        db = pd.read_csv(DBfile)
        return db[['formula']]
    except FileNotFoundError:
        st.error(f"Database file not found at {DBfile}. Please ensure the path is correct.")
        return pd.DataFrame(columns=['formula'])

def show_content(content, color=DEFAULT_COLOR):
    """
    Appends a formatted message to the conversation history in session_state.
    The content will be displayed later by the main rendering loop.
    """
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
    
    # Append a dictionary with color and text to the session state list
    st.session_state.conversation_list.append({"color": color, "text": content.replace("\n", "<br>")})
    # Also print to console for debugging, as in the original app
    print(content)

def problem_conceptualization(input_battery, condition):
    """Generates a prompt based on the current mode and conditions."""
    mode = condition[0]

    if mode == "initial":
        task_index_prompt_template = "We have a Na cathode material FORMULA_PLACEHOLDER. Can you optimize it to develop new cathode materials with higher capacity and improved stability? You can introduce new elements from the following groups: carbon group, alkaline earth metals group, and transition elements, excluding radioactive elements; and incorporate new elements directly into the chemical formula, rather than listing them separately; and give the ratio of each element; and adjust the ratio of existing elements. My requirements are proposing five optimized battery formulations, listing them in bullet points (in asterisk *, not - or number or any other symbol), ensuring each formula is chemically valid and realistic for battery applications, and providing reasoning for each modification."
        prompt = task_index_prompt_template.replace('FORMULA_PLACEHOLDER', input_battery)

    elif mode == "update_with_generated_battery_list":
        generated_battery_list = condition[1]
        prompt = "You generated some existing or invalid battery compositions that need to be replaced with valid ones (one for each).\n"

        not_novel_list = [b for b in generated_battery_list if st.session_state.battery_record[b] == "not novel"]
        invalid_list = [b for b in generated_battery_list if st.session_state.battery_record[b] == "invalid"]
        
        if not_novel_list:
            prompt += "These batteries have been discovered before:\n" + "\n".join(f"* {b}" for b in not_novel_list) + "\n"

        if invalid_list:
            prompt_list = []
            for invalid_battery in invalid_list:
                retrieved_battery = st.session_state.retrieved_battery_record.get(invalid_battery)
                if retrieved_battery:
                    prompt_list.append(f"* {invalid_battery} (a retrieved similar and correct battery is {retrieved_battery})")
                else:
                    prompt_list.append(f"* {invalid_battery}")
            prompt += "These invalid batteries are:\n" + "\n".join(prompt_list) + "\n"

        prompt += "When replacing the invalid or existing compositions, you can replace the newly added elements with elements of lower atomic mass; and adjust the ratio of existing elements; and introduce new elements. The new compositions must be stable and have a higher capacity. The final outputs should include newly generated valid compositions, skip the retrieved batteries, and be listed in bullet points (in asterisk *, not - or number or any other symbol)."
    
    else:
        raise ValueError("Mode should be in [initial, update_with_generated_battery_list].")

    return prompt

def initialize_state():
    """Initializes all necessary variables in st.session_state."""
    st.session_state.conversation_list = []
    st.session_state.LLM_messages = []
    st.session_state.condition_list = []
    st.session_state.input_battery_list = []
    st.session_state.generated_text_list = []
    st.session_state.generated_battery_list = []
    st.session_state.battery_record = defaultdict(str)
    st.session_state.retrieved_battery_record = defaultdict(str)
    st.session_state.main_text_area = "No need to enter prompt."
    st.session_state.initialized = True
    print("===== STATE INITIALIZED =====")

# --- Streamlit App Main Body ---

st.title("🔋 ChatBattery Agent Workflow")
st.markdown("A Streamlit interface for the multi-agent battery material discovery process.")

# --- Load Data ---
retrieval_DB = load_retrieval_DB()

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Controls")
    
    # LLM Selection
    llm_type = st.selectbox(
        'Select LLM Type',
        ("gpt-4.1-mini", "chatgpt_o1", "chatgpt_o3"),
        key='llm_type'
    )

    # Start/Reset Button
    if st.button("Start / Reset Conversation", type="primary"):
        initialize_state()
        show_content("=====" * 10)
        response_message = "[ChatBattery]\nStart editing. Please enter the input battery, and press button 'Step 1.1' to start.\n\n"
        show_content(response_message)
        
        # Set up LLM messages based on selection
        if st.session_state.llm_type in ["gpt-4.1-mini"]:
            st.session_state.LLM_messages = [{"role": "system", "content": "You are an expert in the field of material and chemistry."}]
        else:
            st.session_state.LLM_messages = []
            
        st.session_state.condition_list.append(("initial",))
        st.session_state.main_text_area = "Please enter in your input battery (e.g., Na3V2(PO4)3)."
        st.rerun()

# --- Initialize session state on first run ---
if "initialized" not in st.session_state:
    initialize_state()

# --- Main Page Layout ---
# Area for displaying the conversation
chat_container = st.container(height=500, border=True)
with chat_container:
    for item in st.session_state.conversation_list:
        st.markdown(f'<p style="color:{item["color"]};">{item["text"]}</p>', unsafe_allow_html=True)

# Text area for user input and prompts
main_text = st.text_area("User Input & Prompt Area", value=st.session_state.main_text_area, height=250, key="main_text_input")

# --- Workflow Step Buttons ---
st.markdown("---")
st.subheader("Workflow Steps")
cols = st.columns(3)

# --- Button Logic (Replaces Flask's request.form checks) ---

# Step 1.1: Problem Conceptualization
if cols[0].button("Step 1.1: Conceptualize Problem", use_container_width=True):
    show_content("========== Step 1. Problem Conceptualization ==========")
    condition = st.session_state.condition_list[-1]
    
    if condition[0] == "initial":
        input_battery = main_text.strip()
        st.session_state.input_battery_list.append(input_battery)
    else: # "update_with_generated_battery_list"
        input_battery = st.session_state.input_battery_list[-1]
        generated_battery_list = condition[1]
        valid_list = [b for b in generated_battery_list if st.session_state.battery_record[b] == "valid"]
        if valid_list:
            content = "[ChatBattery]\nThese are the valid batteries from the previous round:\n" + "\n".join(f"* {b}" for b in valid_list) + "\n\n"
            show_content(content)

    prompt = problem_conceptualization(input_battery=input_battery, condition=condition)
    content = f"[Human Agent]\n{prompt}\n\n"
    show_content(content)
    st.session_state.main_text_area = prompt
    st.rerun()

# Step 1.2: Confirm/Edit Prompt (This is now implicit in editing the text_area)
# This button is less necessary in Streamlit, but we can keep it for workflow consistency.
# It effectively just confirms the content of the text area.
if cols[1].button("Step 1.2: Confirm/Edit Prompt", use_container_width=True):
    prompt = main_text.strip()
    content = f"[Human Agent]\n{prompt}\n\n"
    show_content(content) # Show confirmation
    st.session_state.main_text_area = prompt + "\n\nNext double-check or move to Step 2.1."
    st.rerun()

# Step 2.1: Hypothesis Generation (LLM Call)
if cols[2].button("Step 2.1: Generate Hypotheses", use_container_width=True):
    show_content("========== Step 2. Hypothesis Generation ==========")
    content = main_text.replace("Next double-check or move to Step 2.1.", "").strip()
    st.session_state.LLM_messages.append({"role": "user", "content": content})

    # Get valid batteries from previous rounds
    previous_valid_list = []
    if st.session_state.generated_battery_list:
        last_gen_list = st.session_state.generated_battery_list[-1]
        previous_valid_list = [b for b in last_gen_list if st.session_state.battery_record[b] == "valid"]

    # Call LLM
    with st.spinner("LLM is generating hypotheses..."):
        generated_text, new_batteries = LLM_Agent.optimize_batteries(st.session_state.LLM_messages, st.session_state.llm_type)
    
    st.session_state.LLM_messages.append({"role": "assistant", "content": generated_text})
    
    # Combine new and previously valid batteries
    current_generated_list = new_batteries + previous_valid_list
    st.session_state.generated_battery_list.append(current_generated_list)
    
    show_content(f"[LLM Agent]\n{generated_text}\n\n")
    
    # Prepare for next step
    textarea_content = "Please confirm if the following formulas match the LLM's reply (edit if necessary).\n"
    textarea_content += "\n".join(f"* {b}" for b in current_generated_list)
    st.session_state.main_text_area = textarea_content
    st.rerun()


# --- Second Row of Buttons ---
cols2 = st.columns(3)

# Step 2.3: Confirm Parsed Formulas
if cols2[0].button("Step 2.3: Confirm Formulas", use_container_width=True):
    show_content("========== Confirming Parsed Formulas ==========")
    confirmed_batteries = []
    for line in main_text.split("\n"):
        if line.startswith("*"):
            battery = line.replace("*", "").strip()
            if battery:
                confirmed_batteries.append(battery)
    
    st.session_state.generated_battery_list[-1] = confirmed_batteries
    show_content("[Human Agent]\nConfirmed formulas:\n" + "\n".join(f"* {b}" for b in confirmed_batteries) + "\n\n")
    st.session_state.main_text_area = "Formulas confirmed. Next, move to Step 3.1."
    st.rerun()

# Step 3.1: Feasibility Validation (DB Search)
if cols2[1].button("Step 3.1: Validate Feasibility", use_container_width=True):
    show_content("========== Step 3. Hypothesis Feasibility Validation ==========")
    content = "[Search Agent]\n"
    generated_battery_list = st.session_state.generated_battery_list[-1]
    
    with st.spinner("Searching databases for novelty..."):
        for battery in generated_battery_list:
            st.session_state.battery_record[battery] = "novel" # Assume novel initially
            content += f"\n********** searching {battery} in DB **********\n"
            
            # ICSD Search
            exist_icsd = Search_Agent.ICSD_search(battery, retrieval_DB["formula"].tolist())
            content += f"ICSD database: {'exists' if exist_icsd else 'does not exist'}\n"
            if exist_icsd: st.session_state.battery_record[battery] = "not novel"
            
            # MP Search
            exist_mp = Search_Agent.MP_search(battery)
            content += f"Materials Project: {'exists' if exist_mp else 'does not exist'}\n"
            if exist_mp: st.session_state.battery_record[battery] = "not novel"

    show_content(content + "\n\n")
    st.session_state.main_text_area = "Feasibility checked. Next, move to Step 4.1."
    st.rerun()


# Step 4.1: Hypothesis Testing
if cols2[2].button("Step 4.1: Test Hypotheses", use_container_width=True):
    show_content("========== Step 4. Hypothesis Testing ==========")
    input_battery = st.session_state.input_battery_list[-1]
    generated_battery_list = st.session_state.generated_battery_list[-1]

    input_value = Domain_Agent.calculate_theoretical_capacity(input_battery)
    show_content(f"[Domain Agent] Input battery {input_battery} has capacity {input_value:.3f}")
    
    show_content("[Decision Agent]")
    answer_list = Decision_Agent.decide_pairs(input_battery, generated_battery_list)
    
    all_pass = True
    for gen_battery, output_value, is_valid in answer_list:
        novelty_status = "not novel" if st.session_state.battery_record[gen_battery] == "not novel" else "novel"
        
        base_content = f"* Candidate optimized battery {gen_battery} is {novelty_status}"
        if is_valid:
            content = base_content + f" and valid, <span style='color:{DOMAIN_AGENT_COLOR}'>with capacity {output_value:.3f}</span>"
            if novelty_status == "novel":
                st.session_state.battery_record[gen_battery] = "valid"
        else:
            all_pass = False
            content = base_content + f" and invalid, <span style='color:{DOMAIN_AGENT_COLOR}'>with capacity {output_value:.3f}</span>"
            if novelty_status == "novel":
                st.session_state.battery_record[gen_battery] = "invalid"

        show_content(content, color=DECISION_AGENT_COLOR)

        # If novel but invalid, try retrieval
        if novelty_status == "novel" and not is_valid:
            with st.spinner(f"Retrieving similar valid battery for {gen_battery}..."):
                try:
                    ret_battery, ret_capacity = Retrieval_Agent.retrieve_with_domain_feedback(retrieval_DB, input_battery, gen_battery)
                    ret_content = f"[Retrieval Agent] Retrieved battery {ret_battery} <span style='color:{DOMAIN_AGENT_COLOR}'>with capacity {ret_capacity:.3f}</span> is the most similar to the invalid candidate and serves as a valid optimization."
                    st.session_state.retrieved_battery_record[gen_battery] = ret_battery
                except Exception as e:
                    ret_content = f"[Retrieval Agent] No valid battery was retrieved for {gen_battery}. Error: {e}"
                    st.session_state.retrieved_battery_record[gen_battery] = None
            show_content(ret_content)

    if all_pass:
        st.session_state.main_text_area = "✅ Success! All generated batteries are valid. The process is complete."
        st.balloons()
    else:
        condition = ("update_with_generated_battery_list", generated_battery_list)
        st.session_state.condition_list.append(condition)
        st.session_state.main_text_area = "⚠️ Some batteries were invalid. Please return to 'Step 1.1' to refine them."

    st.rerun()
