from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass


def _get_text(root: ET.Element, tag: str) -> str:
    el = root.find(tag)
    return "" if el is None or el.text is None else el.text


@dataclass(frozen=True)
class WeComKfEvent:
    msg_type: str
    event: str
    token: str
    open_kfid: str


def parse_kf_event_xml(xml_bytes: bytes) -> WeComKfEvent:
    """
    Decrypted XML from WeCom WeChat Customer Service callbacks.
    For kf_msg_or_event, it carries a Token used to call sync_msg.
    """
    root = ET.fromstring(xml_bytes)
    return WeComKfEvent(
        msg_type=_get_text(root, "MsgType"),
        event=_get_text(root, "Event"),
        token=_get_text(root, "Token"),
        open_kfid=_get_text(root, "OpenKfId"),
    )

