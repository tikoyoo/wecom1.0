"""
wxAuto 微信自动监听与回复脚本（独立本地运行）

功能：
1. 监听指定联系人/群消息
2. 学生数据查询（今日/本周/作业）→ 调服务器 API
3. 知识库问答 → 本地 BM25 检索 + DeepSeek AI
4. 闲聊自动回复

依赖安装：pip install wxauto requests jieba rank-bm25

使用：
1. 把知识库文件（txt/md）放到 knowledge/ 目录
2. 编辑下方配置
3. 确保微信桌面版已登录
4. python wx_bot.py
"""

import json
import os
import re
import time
import logging
from pathlib import Path

import jieba
import requests
from rank_bm25 import BM25Okapi
from wxauto import WeChat

# ============ 配置 ============

# DeepSeek API
DEEPSEEK_API_KEY = "sk-119f7a1e1c044dea882e2dc113aabd43"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# 服务器地址（用于查询学生数据）
SERVER_URL = "https://wecom.antmaker.vip"

# 本地知识库目录（放 txt/md 文件）
KNOWLEDGE_DIR = "./knowledge"

# 监听的联系人（微信昵称或备注名）
LISTEN_CONTACTS = [
    # "张三妈妈",
    # "李四爸爸",
]

# 监听的群（群名称）
LISTEN_GROUPS = [
    # "CSP-J4班家长群",
]

# 联系人 → 学生 UID 映射（用于查询学生数据）
# key: 微信昵称/备注名, value: Hydro student_uid
CONTACT_STUDENT_MAP = {
    # "张三妈妈": "37",
    # "李四爸爸": "42",
}

# 群聊中机器人名称，群消息需要 @这个名称 才回复；留空则群里所有消息都回复
BOT_NAME = ""

# 轮询间隔（秒）
POLL_INTERVAL = 2

# ============ 日志 ============

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wx-bot")

# ============ 本地知识库 ============


def _tokenize(text: str) -> list[str]:
    return [t.strip() for t in jieba.lcut(text) if t.strip() and len(t.strip()) >= 1]


def _chunk_text(text: str, size: int = 800) -> list[str]:
    """按段落合并切块。"""
    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks, buf = [], ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        candidate = (buf + "\n\n" + para).strip() if buf else para
        if len(candidate) <= size:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            if len(para) > size:
                # 硬切
                start = 0
                while start < len(para):
                    chunks.append(para[start:start + size])
                    start += size - 100
            else:
                buf = para
    if buf:
        chunks.append(buf)
    return chunks or [text[:size]] if text.strip() else []


class LocalKB:
    """本地 BM25 知识库。"""

    def __init__(self):
        self.chunks: list[str] = []
        self.tokenized: list[list[str]] = []
        self.bm25: BM25Okapi | None = None

    def load(self, directory: str):
        """从目录加载所有 txt/md 文件。"""
        p = Path(directory)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            log.info("已创建知识库目录: %s（请放入 txt/md 文件后重启）", p)
            return

        files = list(p.glob("*.txt")) + list(p.glob("*.md"))
        if not files:
            log.info("知识库目录为空: %s", p)
            return

        for f in files:
            try:
                text = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = f.read_text(encoding="gbk", errors="ignore")
            chunks = _chunk_text(text)
            self.chunks.extend(chunks)
            log.info("已加载知识库: %s → %d 个片段", f.name, len(chunks))

        if self.chunks:
            self.tokenized = [_tokenize(c) for c in self.chunks]
            self.bm25 = BM25Okapi(self.tokenized)
            log.info("知识库索引构建完成，共 %d 个片段", len(self.chunks))

    def search(self, query: str, top_k: int = 3, min_score: float = 1.0) -> list[tuple[str, float]]:
        """检索相关片段。"""
        if not self.bm25 or not self.chunks:
            return []
        q = _tokenize(query)
        scores = self.bm25.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = [(self.chunks[i], float(scores[i])) for i in ranked if float(scores[i]) >= min_score]
        return results[:top_k]


kb = LocalKB()

# ============ DeepSeek API ============


def chat_with_ai(question: str, kb_context: str = "", history: list[dict] | None = None) -> str:
    """调用 DeepSeek API。"""
    if not DEEPSEEK_API_KEY:
        return "未配置 DEEPSEEK_API_KEY。"

    system = (
        "你是编程教育机构的客服助手。\n"
        "【规则】\n"
        "1. 只能根据下方【知识库】内容回答，不得编造或推测知识库以外的信息。\n"
        "2. 如果知识库中没有相关内容，必须回复：'抱歉，我暂时没有这方面的信息，建议您直接联系老师确认。'\n"
        "3. 回答简洁、中文、结构清晰，不超过300字。\n"
        "4. 不要重复用户的问题，直接给出答案。"
    )
    if not kb_context:
        system += "\n\n注意：当前知识库无相关内容，如无法确定请如实说明，不要编造。"

    messages = [{"role": "system", "content": system}]
    if kb_context:
        messages.append({"role": "system", "content": f"【知识库】\n{kb_context}"})
    if history:
        messages.extend(history[-6:])  # 最近 3 轮
    messages.append({"role": "user", "content": question})

    try:
        url = DEEPSEEK_BASE_URL.rstrip("/") + "/chat/completions"
        resp = requests.post(
            url,
            json={"model": DEEPSEEK_MODEL, "messages": messages, "temperature": 0.2},
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            timeout=40,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.exception("DeepSeek API 调用失败")
        return f"AI 服务暂时不可用：{e}"


# ============ 学生数据查询（调服务器 API）============


def query_student_stats(student_uid: str) -> dict | None:
    """调服务器 API 查询学生 Hydro 数据。"""
    try:
        url = f"{SERVER_URL}/api/h5/student-stats-data"
        resp = requests.get(url, params={"student_uid": student_uid}, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning("查询学生数据失败: uid=%s err=%s", student_uid, e)
        return None


def format_today_reply(stats: dict) -> str:
    name = stats.get("name", "")
    uid = stats.get("uid", "")
    today_ac = int(stats.get("today_ac") or 0)
    today_submits = int(stats.get("today_submits") or 0)
    today_pids = stats.get("today_ac_pids") or []
    pids_text = "、".join(str(p) for p in today_pids) if today_pids else "（无）"
    return (
        f"📊 {name} 今日做题情况\n\n"
        f"✅ 今日 AC：{today_ac} 题\n"
        f"📝 今日提交：{today_submits} 次\n"
        f"📋 AC 题号：{pids_text}"
    )


def format_week_reply(stats: dict) -> str:
    name = stats.get("name", "")
    week_ac = int(stats.get("week_ac_count") or 0)
    week_pids = stats.get("week_ac_pids") or []
    hw_title = stats.get("hw_title") or "暂无"
    hw_tasks = stats.get("hw_tasks") or []
    hw_done = sum(1 for t in hw_tasks if t.get("ac"))
    hw_total = len(hw_tasks)
    pids_text = "、".join(str(p) for p in week_pids) if week_pids else "（无）"
    return (
        f"📊 {name} 本周做题情况\n\n"
        f"💡 本周 AC：{week_ac} 题\n"
        f"📋 AC 题号：{pids_text}\n\n"
        f"📚 当前作业：{hw_title}\n"
        f"✅ 完成情况：{hw_done}/{hw_total}"
    )


def format_hw_reply(stats: dict) -> str:
    name = stats.get("name", "")
    hw_title = stats.get("hw_title") or ""
    hw_tasks = stats.get("hw_tasks") or []
    if not hw_title:
        return f"{name} 当前暂无进行中的作业。"
    hw_done = sum(1 for t in hw_tasks if t.get("ac"))
    hw_total = len(hw_tasks)
    done_list = [str(t.get("pid", "")) for t in hw_tasks if t.get("ac")]
    undone_list = [str(t.get("pid", "")) for t in hw_tasks if not t.get("ac")]
    done_emoji = "✅" if hw_done >= hw_total and hw_total > 0 else "⏳"
    lines = [
        f"📚 {name} 作业完成情况\n",
        f"作业：{hw_title}",
        f"{done_emoji} 完成：{hw_done}/{hw_total} 题",
    ]
    if done_list:
        lines.append(f"✅ 已完成：{'、'.join(done_list)}")
    if undone_list:
        lines.append(f"⬜ 未完成：{'、'.join(undone_list)}")
    return "\n".join(lines)


# ============ 意图识别 ============

_CHITCHAT_PATTERNS = [
    r"^(你好|hi|hello|在吗|在不|哈喽|嗨|hey)[\s？?！!。~～]*$",
    r"^(谢谢|感谢|好的|收到|明白|了解|ok|okay|好)[\s！!。~～]*$",
    r"^(再见|拜拜|bye|goodbye|晚安|88)[\s！!。~～]*$",
    r"^(哈哈|哈哈哈|嗯嗯|嗯|哦|哦哦|噢|好的好的)[\s！!。~～]*$",
]

HELP_TEXT = (
    "您好！我是学习助手，可以帮您查询孩子的学习情况。\n\n"
    "支持以下查询：\n"
    "• 发送「今日」或「今天」— 查看今日做题情况\n"
    "• 发送「本周」或「这周」— 查看本周做题情况\n"
    "• 发送「作业」— 查看当前作业完成情况\n\n"
    "其他问题也可以直接发送，我会尽力解答。"
)


def is_chitchat(text: str) -> bool:
    for p in _CHITCHAT_PATTERNS:
        if re.match(p, text.strip(), re.IGNORECASE):
            return True
    return False


def match_intent(text: str) -> str:
    """识别意图：today / week / hw / unknown。"""
    t = text.strip()
    if re.search(r"今[天日]|今日|查今", t):
        return "today"
    if re.search(r"本周|这周|一周|周报|查周", t):
        return "week"
    if re.search(r"作业|homework|hw", t, re.IGNORECASE):
        return "hw"
    return "unknown"


# ============ 对话记忆 ============

# sender -> 最近消息列表
_chat_history: dict[str, list[dict]] = {}
MAX_HISTORY = 6  # 保留最近 3 轮


def _get_history(sender: str) -> list[dict]:
    return _chat_history.get(sender, [])


def _add_history(sender: str, role: str, content: str):
    if sender not in _chat_history:
        _chat_history[sender] = []
    _chat_history[sender].append({"role": role, "content": content})
    if len(_chat_history[sender]) > MAX_HISTORY:
        _chat_history[sender] = _chat_history[sender][-MAX_HISTORY:]


# ============ 消息处理核心 ============


def handle_message(sender: str, content: str) -> str:
    """处理一条消息，返回回复文本。"""
    content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", content).strip()
    if not content:
        return ""

    # 帮助
    if content in ("帮助", "help", "?", "？", "菜单"):
        return HELP_TEXT

    # 闲聊
    if is_chitchat(content):
        for kw, reply in [("谢", "不客气！"), ("再见", "再见！"), ("拜", "拜拜！")]:
            if kw in content:
                return reply
        return "您好！请问有什么可以帮您？发送「帮助」查看功能。"

    # 学生数据查询
    student_uid = CONTACT_STUDENT_MAP.get(sender, "")
    intent = match_intent(content)

    if intent != "unknown" and student_uid:
        stats = query_student_stats(student_uid)
        if not stats or stats.get("error"):
            return "暂时无法获取数据，请稍后再试。"
        if intent == "today":
            return format_today_reply(stats)
        elif intent == "week":
            return format_week_reply(stats)
        elif intent == "hw":
            return format_hw_reply(stats)

    if intent != "unknown" and not student_uid:
        return "暂未绑定学生信息，无法查询做题数据。请联系老师完成绑定。"

    # 知识库检索 + AI 回复
    hits = kb.search(content)
    kb_context = ""
    if hits:
        lines = [f"[来源{i+1}] {text.strip()}" for i, (text, score) in enumerate(hits)]
        kb_context = "\n\n".join(lines)

    history = _get_history(sender)
    reply = chat_with_ai(content, kb_context=kb_context, history=history)

    _add_history(sender, "user", content)
    _add_history(sender, "assistant", reply)

    return reply


# ============ 群消息判断 ============


def should_reply_group_msg(content: str) -> tuple[bool, str]:
    if not BOT_NAME:
        return True, content.strip()
    at_tag = f"@{BOT_NAME}"
    if at_tag in content:
        return True, content.replace(at_tag, "").strip()
    return False, ""


# ============ 主循环 ============


def main():
    # 加载知识库
    kb.load(KNOWLEDGE_DIR)

    log.info("正在连接微信桌面版...")
    wx = WeChat()
    log.info("微信连接成功")

    listen_list = LISTEN_CONTACTS + LISTEN_GROUPS
    if not listen_list:
        log.warning("未配置监听对象，请编辑 LISTEN_CONTACTS 或 LISTEN_GROUPS")
        return

    for name in listen_list:
        wx.AddListenChat(who=name)
        log.info("已添加监听: %s", name)

    log.info("开始监听消息，按 Ctrl+C 停止...")

    while True:
        try:
            msgs = wx.GetListenMessage()
            if not msgs:
                time.sleep(POLL_INTERVAL)
                continue

            for chat, msg_list in msgs.items():
                chat_name = chat.who if hasattr(chat, "who") else str(chat)
                is_group = chat_name in LISTEN_GROUPS

                for msg in msg_list:
                    if msg.type != "friend" or not msg.content:
                        continue

                    sender = msg.sender if hasattr(msg, "sender") else chat_name
                    content = msg.content.strip()
                    if not content:
                        continue

                    if is_group:
                        should, clean = should_reply_group_msg(content)
                        if not should:
                            continue
                        content = clean
                        if not content:
                            continue

                    log.info("收到: [%s] %s: %s", chat_name, sender, content[:100])

                    reply = handle_message(sender, content)
                    if not reply:
                        continue

                    if is_group and sender:
                        reply = f"@{sender} {reply}"

                    chat.SendMsg(reply)
                    log.info("回复: [%s] len=%d", chat_name, len(reply))

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            log.info("已停止")
            break
        except Exception:
            log.exception("异常，继续运行...")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
