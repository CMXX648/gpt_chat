import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 实例化每个API提供者的客户端
qwen_client = OpenAI(api_key=os.getenv('DASHSCOPE_API_KEY'), base_url=os.getenv('QWEN_API_BASE'))


def perform_chat(client, model_name, messages, functions, temperature, max_tokens):
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        functions=functions
    )
    return response.choices[0].message.content


def qwen_chat(system_prompt, user_content, functions=None, max_tokens=2000):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    response = perform_chat(qwen_client, "qwen-plus", messages, functions=functions, temperature=0.80,
                            max_tokens=max_tokens)
    return response
