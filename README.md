# Coding Agent —— 从零手写的编码 Agent

一个用 Python 从零实现的本地编码 Agent，不依赖任何 Agent 框架。采用 ReAct（思考→行动→观察）循环，接入 DeepSeek 大模型（OpenAI 兼容协议），可自主完成编程任务。

## Git 仓库地址
https://github.com/3z-c/zc_coding-agent.git

## 如何运行

1. 克隆仓库：`git clone https://github.com/3z-c/zc_coding-agent.git`
2. 安装依赖：`pip install -r requirements.txt`
3. 配置密钥：复制 `.env.example` 为 `.env`，填入 `DEEPSEEK_API_KEY`
4. 运行：
   - 单次任务：`python main.py "你的任务"`
   - 交互模式：`python main.py`（连续对话，输入 exit 退出）
   - 可选参数：`--steps` 限制单次任务步数，`--cwd` 指定工作目录

## 特色功能

- **ReAct 主循环**：思考→行动→观察，模型自主调用工具直到完成任务；LLM 调用失败自动指数退避重试。
- **六大内置工具**：读文件、写文件、列目录、精确替换、删除文件、执行命令；路径限制在工作目录内，防止越界。
- **错误自我修正**：工具执行失败、参数解析失败都会把错误回填给模型，让模型读错误自行修正，不静默吞掉。
- **上下文管理**：工具结果分级（小结果保留、大结果截断、超大结果落盘只留预览指针）+ token 预算裁剪，长任务不爆上下文。
- **交互式多轮**：同一会话保留历史，Agent 记得此前说过、做过什么，可连续演进。

## 其它说明

- 仅依赖 `openai` 官方 SDK 与 `python-dotenv`；LLM 适配层与工具层解耦，换厂商只需改 `llm/client.py`。
