import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class ChatBot:
    """
    基于 DeepSeek API 的聊天机器人。
    提供简单的聊天界面和 API 调用功能。
    """

    def __init__(self, api_key: str = None, model: str = "deepseek-chat",
                 api_url: str = "https://api.deepseek.com/v1"):
        """
        使用 API 配置初始化聊天机器人。

        参数:
            api_key: DeepSeek API 密钥，如果为 None 则从环境变量中读取
            model: 使用的模型名称
            api_url: API 服务的基础 URL
        """
        # 如果未提供 API 密钥，则从环境变量中读取
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("未提供 API 密钥。请设置 DEEPSEEK_API_KEY 环境变量或在构造函数中提供。")
            
        self.model = model
        self.api_url = api_url
        self._client = None
        self.conversation_history = []

    @property
    def client(self):
        """API 客户端的懒加载初始化。"""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_url
            )
        return self._client

    def chat(self, user_message: str, system_prompt: str = None, temperature: float = 0.7):
        """
        发送消息到 DeepSeek API 并获取回复。

        参数:
            user_message: 用户输入的消息
            system_prompt: 可选的系统提示
            temperature: 温度参数，控制回复的随机性

        返回:
            str: AI 的回复
        """
        # 准备消息
        messages = []
        
        # 添加系统提示（如果有）
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 添加历史对话
        messages.extend(self.conversation_history)
        
        # 添加用户消息
        messages.append({"role": "user", "content": user_message})
        
        try:
            # 调用 API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=2000
            )
            
            # 获取回复
            if response.choices:
                ai_message = response.choices[0].message.content
                
                # 更新对话历史
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": ai_message})
                
                return ai_message
            else:
                return "抱歉，我无法生成回复。"
            
        except Exception as e:
            return f"API 调用出错: {e}"

    def clear_history(self):
        """清除对话历史"""
        self.conversation_history = []


def main():
    """主函数，运行聊天界面"""
    # 初始化聊天机器人
    chatbot = ChatBot()
    
    print("欢迎使用 DeepSeek 聊天机器人！输入 'exit' 退出，输入 'clear' 清除对话历史。")
    
    while True:
        # 获取用户输入
        user_input = input("\n你: ")
        
        # 检查退出命令
        if user_input.lower() == 'exit':
            print("再见！")
            break
        
        # 检查清除历史命令
        if user_input.lower() == 'clear':
            chatbot.clear_history()
            print("对话历史已清除。")
            continue
        
        # 获取 AI 回复
        response = chatbot.chat(user_input)
        
        # 显示回复
        print(f"\nAI: {response}")


if __name__ == "__main__":
    main()