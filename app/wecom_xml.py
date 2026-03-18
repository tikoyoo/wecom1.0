from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass


def _get_text(root: ET.Element, tag: str) -> str:
    el = root.find(tag)
    return "" if el is None or el.text is None else el.text


@dataclass(frozen=True)
class WeComEncryptedIncoming:
    to_user_name: str
    encrypt: str


def parse_encrypted_xml(xml_bytes: bytes) -> WeComEncryptedIncoming:
    root = ET.fromstring(xml_bytes)
    return WeComEncryptedIncoming(
        to_user_name=_get_text(root, "ToUserName"),
        encrypt=_get_text(root, "Encrypt"),
    )


@dataclass(frozen=True)
class WeComTextMessage:
    to_user_name: str
    from_user_name: str
    msg_type: str
    content: str
    msg_id: str
    create_time: int


def parse_plain_xml(xml_bytes: bytes) -> WeComTextMessage:
    root = ET.fromstring(xml_bytes)
    return WeComTextMessage(
        to_user_name=_get_text(root, "ToUserName"),
        from_user_name=_get_text(root, "FromUserName"),
        msg_type=_get_text(root, "MsgType"),
        content=_get_text(root, "Content"),
        msg_id=_get_text(root, "MsgId"),
        create_time=int(_get_text(root, "CreateTime") or "0"),
    )


def build_plain_text_reply(to_user: str, from_user: str, content: str) -> bytes:
    now = int(time.time())
    xml = f"""<xml>
<ToUserName><![CDATA[{to_user}]]></ToUserName>
<FromUserName><![CDATA[{from_user}]]></FromUserName>
<CreateTime>{now}</CreateTime>
<MsgType><![CDATA[text]]></MsgType>
<Content><![CDATA[{content}]]></Content>
</xml>"""
    return xml.encode("utf-8")


def build_encrypted_reply_xml(encrypt: str, signature: str, timestamp: str, nonce: str) -> bytes:
    xml = f"""<xml>
<Encrypt><![CDATA[{encrypt}]]></Encrypt>
<MsgSignature><![CDATA[{signature}]]></MsgSignature>
<TimeStamp>{timestamp}</TimeStamp>
<Nonce><![CDATA[{nonce}]]></Nonce>
</xml>"""
    return xml.encode("utf-8")

