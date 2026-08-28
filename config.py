"""统一配置加载。

把散落在各脚本里的"读 .env"逻辑收敛到这一处。整个项目只有一个入口读配置。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录(coding-agent/)
PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Config:
    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_env(cls) -> "Config":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key or api_key.startswith("sk-xxx"):
            raise ValueError("未配置 DEEPSEEK_API_KEY，请在项目根目录 .env 中填入真实密钥")
        return cls(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
        )
