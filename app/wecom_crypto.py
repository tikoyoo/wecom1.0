from __future__ import annotations

import base64
import os
import struct
from dataclasses import dataclass
from hashlib import sha1

from Crypto.Cipher import AES


def _pkcs7_pad(data: bytes, block_size: int = 32) -> bytes:
    pad = block_size - (len(data) % block_size)
    return data + bytes([pad]) * pad


def _pkcs7_unpad(data: bytes) -> bytes:
    if not data:
        raise ValueError("empty data")
    pad = data[-1]
    if pad < 1 or pad > 32:
        raise ValueError("bad padding")
    return data[:-pad]


def _sha1_signature(token: str, timestamp: str, nonce: str, encrypt: str) -> str:
    s = "".join(sorted([token, timestamp, nonce, encrypt]))
    return sha1(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WeComCrypto:
    token: str
    encoding_aes_key: str
    corp_id: str

    def __post_init__(self):
        aes_key = base64.b64decode(self.encoding_aes_key + "=")
        if len(aes_key) != 32:
            raise ValueError("encoding_aes_key must decode to 32 bytes")

    @property
    def _aes_key(self) -> bytes:
        return base64.b64decode(self.encoding_aes_key + "=")

    def verify_signature(self, msg_signature: str, timestamp: str, nonce: str, encrypt: str) -> bool:
        return msg_signature == _sha1_signature(self.token, timestamp, nonce, encrypt)

    def decrypt(self, encrypt_b64: str) -> bytes:
        cipher = AES.new(self._aes_key, AES.MODE_CBC, iv=self._aes_key[:16])
        plain_padded = cipher.decrypt(base64.b64decode(encrypt_b64))
        plain = _pkcs7_unpad(plain_padded)

        # 16 random + 4 bytes length + xml + corp_id
        if len(plain) < 20:
            raise ValueError("plain too short")
        msg_len = struct.unpack("!I", plain[16:20])[0]
        xml = plain[20 : 20 + msg_len]
        recv_corp_id = plain[20 + msg_len :].decode("utf-8")
        if recv_corp_id != self.corp_id:
            raise ValueError("corp_id mismatch")
        return xml

    def encrypt(self, xml: bytes, nonce: str, timestamp: str | None = None) -> tuple[str, str, str]:
        if timestamp is None:
            timestamp = str(int(__import__("time").time()))

        random16 = os.urandom(16)
        msg_len = struct.pack("!I", len(xml))
        plain = random16 + msg_len + xml + self.corp_id.encode("utf-8")
        plain_padded = _pkcs7_pad(plain, 32)

        cipher = AES.new(self._aes_key, AES.MODE_CBC, iv=self._aes_key[:16])
        encrypt = base64.b64encode(cipher.encrypt(plain_padded)).decode("utf-8")
        signature = _sha1_signature(self.token, timestamp, nonce, encrypt)
        return encrypt, signature, timestamp

