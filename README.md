# 企业微信客服机器人（DeepSeek + 知识库 + 多轮记忆 + 管理后台）

目标：把你已跑通的企业微信消息能力，升级成**企业知识库问答**，支持**多轮记忆**，并允许你通过后台**主动给用户推送消息**。

## 功能清单（按你需求对齐）

- 企业微信回调：URL 验证 + 安全模式 AES 解密/加密 + 被动回复
- 主动发消息：用 `corp_id/corp_secret` 获取 `access_token`，调用消息发送接口推送给指定用户
- 知识库：上传文档（<=100 条），切分入库，检索召回（Chroma），回答时引用来源片段
- 多轮：按用户维度保存历史对话，生成答案时带上最近 N 轮
- 大模型：DeepSeek（OpenAI-compatible）
- 管理后台：查看会话/知识库、上传/删除文档、对用户发消息

## 本地启动（Windows PowerShell）

```powershell
cd C:\Users\86182\wecom-customer-service-bot
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

打开：
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`
- 管理后台：`http://127.0.0.1:8000/admin/`

## 上线到云服务器（建议）

- 用 `systemd` 托管进程（自启、崩溃拉起、统一日志）
- 用内网穿透/公网域名 + HTTPS 配置企业微信回调 URL

### systemd 示例（Ubuntu/Debian）

假设你的代码放在 `/opt/wecom-customer-service-bot`，Python venv 在 `.venv`。

1) 创建服务文件 `/etc/systemd/system/wecom-bot.service`

```ini
[Unit]
Description=WeCom KB Bot
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/wecom-customer-service-bot
EnvironmentFile=/opt/wecom-customer-service-bot/.env
ExecStart=/opt/wecom-customer-service-bot/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
```

2) 启动并设为开机自启

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now wecom-bot
sudo systemctl status wecom-bot
```

3) 看日志

```bash
sudo journalctl -u wecom-bot -f
```


