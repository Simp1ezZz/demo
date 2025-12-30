"""
LangChain Demo - 使用 HuggingFace 本地模型
展示 LangChain 1.2+ 的核心功能，无需 API Key
"""
import os
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# 选择要使用的模型（可以修改为其他 HuggingFace 模型）
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # 小模型，适合快速测试
# MODEL_NAME = "THUDM/chatglm3-6b"  # 中文效果更好的模型
# MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # 中等大小的模型

# 全局 LLM 对象
llm = None

def init_model():
    """初始化 HuggingFace 模型"""
    global llm
    
    print(f"📦 正在加载模型 {MODEL_NAME}...")
    print("⚠️  首次运行会下载模型，可能需要一些时间。\n")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # 根据设备选择加载方式
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  使用设备: {device}\n")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    
    if device == "cpu":
        model = model.to(device)
    
    # 创建 pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        return_full_text=False,
        device=0 if device == "cuda" else -1
    )
    
    # 创建 LangChain LLM
    llm = HuggingFacePipeline(pipeline=pipe)
    print("✅ 模型加载完成！\n")


def demo_basic_llm():
    """演示基本的 LLM 调用"""
    print("=" * 50)
    print("演示 1: 基本的 LLM 调用")
    print("=" * 50)
    
    # 简单调用
    response = llm.invoke("用一句话解释什么是人工智能？")
    print(f"回答: {response}\n")


def demo_prompt_template():
    """演示使用提示模板"""
    print("=" * 50)
    print("演示 2: 使用提示模板")
    print("=" * 50)
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_template(
        "你是一个专业的技术顾问，擅长用简单易懂的方式解释技术概念。\n\n请解释一下 {technology} 是什么，并给出一个实际应用例子。"
    )
    
    # 格式化提示
    formatted_prompt = prompt.format(technology="区块链")
    response = llm.invoke(formatted_prompt)
    print(f"回答: {response}\n")


def demo_chain():
    """演示链式调用"""
    print("=" * 50)
    print("演示 3: 链式调用 (LLMChain)")
    print("=" * 50)
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_template(
        "将以下文本翻译成英文，并总结其主要内容：\n\n{text}"
    )
    
    # 创建链
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # 执行链
    result = chain.invoke({
        "text": "人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。"
    })
    
    print(f"结果: {result['text']}\n")


def demo_multi_step():
    """演示多步骤处理"""
    print("=" * 50)
    print("演示 4: 多步骤处理")
    print("=" * 50)
    
    # 步骤1: 生成主题
    step1_prompt = ChatPromptTemplate.from_template(
        "基于以下关键词生成一个技术主题：{keywords}"
    )
    step1_chain = LLMChain(llm=llm, prompt=step1_prompt)
    
    # 步骤2: 基于主题生成内容
    step2_prompt = ChatPromptTemplate.from_template(
        "为主题 '{topic}' 写一段简短的介绍（100字以内）"
    )
    step2_chain = LLMChain(llm=llm, prompt=step2_prompt)
    
    # 执行多步骤
    print("\n正在执行步骤1: 生成主题...")
    topic_result = step1_chain.invoke({"keywords": "机器学习, 深度学习, 神经网络"})
    topic = topic_result['text'].strip()
    
    print("正在执行步骤2: 生成主题介绍...")
    content_result = step2_chain.invoke({"topic": topic})
    
    print(f"\n生成的主题: {topic}")
    print(f"主题介绍: {content_result['text']}\n")


def demo_conversation():
    """演示对话式交互"""
    print("=" * 50)
    print("演示 5: 对话式交互")
    print("=" * 50)
    
    # 构建对话提示
    conversation_history = "你是一个友好的AI助手，擅长回答技术问题。\n\n"
    
    # 第一轮对话
    print("\n第一轮对话:")
    user_input1 = "什么是 LangChain？"
    prompt1 = conversation_history + f"用户: {user_input1}\n助手:"
    response1 = llm.invoke(prompt1)
    print(f"用户: {user_input1}")
    print(f"助手: {response1}\n")
    
    # 第二轮对话（带上下文）
    print("第二轮对话（带上下文）:")
    conversation_history += f"用户: {user_input1}\n助手: {response1}\n\n"
    user_input2 = "它有什么主要优势？"
    prompt2 = conversation_history + f"用户: {user_input2}\n助手:"
    response2 = llm.invoke(prompt2)
    print(f"用户: {user_input2}")
    print(f"助手: {response2}\n")


def main():
    """主函数"""
    print("\n" + "=" * 50)
    print("LangChain 1.2+ Demo - HuggingFace 本地模型")
    print("=" * 50 + "\n")
    
    try:
        # 初始化模型
        init_model()
        
        # 运行各个演示
        demo_basic_llm()
        demo_prompt_template()
        demo_chain()
        demo_multi_step()
        demo_conversation()
        
        print("=" * 50)
        print("所有演示完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        print("\n请确保：")
        print("1. 已安装所有依赖: pip install -r requirements.txt")
        print("2. 网络连接正常（首次运行需要下载模型）")
        print("3. 有足够的磁盘空间（模型文件可能较大）")
        print(f"4. 当前使用的模型: {MODEL_NAME}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
