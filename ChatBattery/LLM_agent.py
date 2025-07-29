import sys
import openai
import time
import google.generativeai as genai
import os


def parse(raw_text, history_battery_list):
    record = []
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if line.startswith("Assistant:"):
            line = line.replace("Assistant:", "").strip()
        if not line.startswith("*"):
            continue
        line = line.split(" ")
        
        output_formula_list = []
        for word in line:
            upper_alpha_count, digit_count = 0, 0
            for char in word:
                if char.isupper():
                    upper_alpha_count += 1
                if char.isdigit():
                    digit_count += 1
            if upper_alpha_count > 1 and digit_count >= 1:
                if not word[-1].isalnum() and word[-1] != ")" and word[-1] != "]":
                    word = word[:-1]
                output_formula_list.append(word)

        output_formula_list = [x for x in output_formula_list if x not in history_battery_list]
        if len(output_formula_list) > 0:
            record.extend(output_formula_list)
    return record


class LLM_Agent:
    @staticmethod
    def optimize_batteries(messages, LLM_type, temperature=0, loaded_model=None, loaded_tokenizer=None):
        print("===== Checking messages in the LLM agent =====")
        for message in messages:
            content = message["content"]
            print("content: {}".format(content))
        print("===== Done checking. =====\n")

        if LLM_type == 'chatgpt_3.5':
            return LLM_Agent.optimize_batteries_chatgpt(messages, model="gpt-3.5-turbo", temperature=temperature)
        elif LLM_type == 'chatgpt_o1':
            return LLM_Agent.optimize_batteries_chatgpt(messages, model="o1-mini", temperature=temperature)
        elif LLM_type == 'chatgpt_o3':
            return LLM_Agent.optimize_batteries_chatgpt(messages, model="o3-mini", temperature=temperature)
        elif LLM_type == 'chatgpt_4o':
            return LLM_Agent.optimize_batteries_chatgpt(messages, model="gpt-4o-mini", temperature=temperature)
        elif LLM_type == 'gemini_2.5_flash':
            return LLM_Agent.optimize_batteries_gemini(messages, model="gemini-2.5-flash", temperature=temperature)
        elif LLM_type == 'gemini_2.5_pro':
            return LLM_Agent.optimize_batteries_gemini(messages, model="gemini-2.5-pro", temperature=temperature)
        elif LLM_type == 'gemini_2.0_flash':
            return LLM_Agent.optimize_batteries_gemini(messages, model="gemini-2.0-flash-exp", temperature=temperature)
        elif LLM_type == "llama2":
            return LLM_Agent.optimize_batteries_open_source(messages, loaded_model=loaded_model, loaded_tokenizer=loaded_tokenizer)
        elif LLM_type == "llama3":
            return LLM_Agent.optimize_batteries_open_source(messages, loaded_model=loaded_model, loaded_tokenizer=loaded_tokenizer)
        elif LLM_type == "qwen":
            return LLM_Agent.optimize_batteries_open_source(messages, loaded_model=loaded_model, loaded_tokenizer=loaded_tokenizer)
        else:
            raise NotImplementedError

    @staticmethod
    def optimize_batteries_chatgpt(messages, model, temperature):
        received = False

        history_battery_list = []

        while not received:
            try:
                if model == "chatgpt_3.5":
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        frequency_penalty=0.2,
                        n=None)
                else:
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        n=None)
                raw_generated_text = response["choices"][0]["message"]['content']

                # ignore batteries that have shown up before
                generated_battery_list = parse(raw_generated_text, history_battery_list)

                print("===== Parsing messages in the LLM agent =====")
                print("raw_generated_text", raw_generated_text.replace("\n", "\t"))
                print("generated_battery_list", generated_battery_list)
                print("===== Done parsing. =====\n")

                if len(generated_battery_list) == 0:
                    raw_generated_battery_list = parse(raw_generated_text, [])
                    print("raw_generated_battery_list", raw_generated_battery_list)
                    messages[-1]["content"] += "Please do not generate batteries in this list {}.".format(raw_generated_battery_list)

                assert len(generated_battery_list) > 0, "The generated batteries have been discussed in our previous rounds of discussion. Will retry."

                received = True
                    
            except Exception as e:
                print(e)
                time.sleep(1)
        return raw_generated_text, generated_battery_list

    @staticmethod
    def optimize_batteries_open_source(messages, loaded_model, loaded_tokenizer):
        received = False
        history_battery_list = []

        input_text = ""
        for message in messages:
            content = message["content"]
            input_text = "{}\n{}".format(input_text, content)
        print("input_text", input_text)
    
        inputs = loaded_tokenizer(input_text, return_tensors="pt").to("cuda")

        while not received:
            try:
                # raw_generated_text = response["choices"][0]["message"]['content']
                outputs = loaded_model.generate(**inputs)
                raw_generated_text = loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)


                # ignore batteries that have shown up before
                generated_battery_list = parse(raw_generated_text, history_battery_list)

                print("===== Parsing messages in the LLM agent =====")
                print("raw_generated_text", raw_generated_text.replace("\n", "\t"))
                print("generated_battery_list", generated_battery_list)
                print("===== Done parsing. =====\n")

                if len(generated_battery_list) == 0:
                    raw_generated_battery_list = parse(raw_generated_text, [])
                    print("raw_generated_battery_list", raw_generated_battery_list)
                    messages[-1]["content"] += "Please do not generate batteries in this list {}.".format(raw_generated_battery_list)

                assert len(generated_battery_list) > 0, "The generated batteries have been discussed in our previous rounds of discussion. Will retry."

                received = True
                    
            except Exception as e:
                print(e)
                time.sleep(1)
        return raw_generated_text, generated_battery_list


    @staticmethod
    def rank_batteries(messages, LLM_type, temperature):
        if LLM_type == 'chatgpt_3.5':
            return LLM_Agent.rank_batteries_chatgpt(messages, model="gpt-3.5-turbo", temperature=temperature)
        elif LLM_type == 'chatgpt_o1':
            return LLM_Agent.rank_batteries_chatgpt(messages, model="o1-mini", temperature=temperature)
        elif LLM_type == 'chatgpt_o3':
            return LLM_Agent.rank_batteries_chatgpt(messages, model="o3-mini", temperature=temperature)
        elif LLM_type == 'chatgpt_4o':
            return LLM_Agent.rank_batteries_chatgpt(messages, model="gpt-4o-mini", temperature=temperature)
        elif LLM_type == 'gemini_2.5_flash':
            return LLM_Agent.rank_batteries_gemini(messages, model="gemini-2.5-flash", temperature=temperature)
        elif LLM_type == 'gemini_2.5_pro':
            return LLM_Agent.rank_batteries_gemini(messages, model="gemini-2.5-pro", temperature=temperature)
        elif LLM_type == 'gemini_2.0_flash':
            return LLM_Agent.rank_batteries_gemini(messages, model="gemini-2.0-flash-exp", temperature=temperature)
        else:
            raise NotImplementedError

    @staticmethod
    def rank_batteries_chatgpt(messages, model, temperature):
        received = False

        if model == "chatgpt_3.5":
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                frequency_penalty=0.2,
                n=None)
        else:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                n=None)
        raw_generated_text = response["choices"][0]["message"]['content']

        return raw_generated_text

    @staticmethod
    def optimize_batteries_gemini(messages, model, temperature):
        # Configure Gemini API
        gemini_api_key = os.getenv('GOOGLE_API_KEY')
        if not gemini_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=gemini_api_key)
        
        received = False
        history_battery_list = []

        while not received:
            try:
                # Create the Gemini model
                gemini_model = genai.GenerativeModel(model)
                
                # Convert OpenAI-style messages to Gemini format
                # For Gemini, we need to combine the messages into a single prompt
                prompt_text = ""
                for message in messages:
                    role = message["role"]
                    content = message["content"]
                    if role == "system":
                        prompt_text += f"System: {content}\n\n"
                    elif role == "user":
                        prompt_text += f"User: {content}\n\n"
                    elif role == "assistant":
                        prompt_text += f"Assistant: {content}\n\n"
                
                # Generate response
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=1000,
                )
                
                response = gemini_model.generate_content(
                    prompt_text,
                    generation_config=generation_config
                )
                
                raw_generated_text = response.text

                # ignore batteries that have shown up before
                generated_battery_list = parse(raw_generated_text, history_battery_list)

                print("===== Parsing messages in the LLM agent =====")
                print("raw_generated_text", raw_generated_text.replace("\n", "\t"))
                print("generated_battery_list", generated_battery_list)
                print("===== Done parsing. =====\n")

                if len(generated_battery_list) == 0:
                    raw_generated_battery_list = parse(raw_generated_text, [])
                    print("raw_generated_battery_list", raw_generated_battery_list)
                    messages[-1]["content"] += "Please do not generate batteries in this list {}.".format(raw_generated_battery_list)

                assert len(generated_battery_list) > 0, "The generated batteries have been discussed in our previous rounds of discussion. Will retry."

                received = True
                    
            except Exception as e:
                print(e)
                time.sleep(1)
        return raw_generated_text, generated_battery_list

    @staticmethod
    def rank_batteries_gemini(messages, model, temperature):
        # Configure Gemini API
        gemini_api_key = os.getenv('GOOGLE_API_KEY')
        if not gemini_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=gemini_api_key)
        
        # Create the Gemini model
        gemini_model = genai.GenerativeModel(model)
        
        # Convert OpenAI-style messages to Gemini format
        prompt_text = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt_text += f"System: {content}\n\n"
            elif role == "user":
                prompt_text += f"User: {content}\n\n"
            elif role == "assistant":
                prompt_text += f"Assistant: {content}\n\n"
        
        # Generate response
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=500,
        )
        
        response = gemini_model.generate_content(
            prompt_text,
            generation_config=generation_config
        )
        
        raw_generated_text = response.text

        return raw_generated_text