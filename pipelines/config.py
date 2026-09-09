from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    model: str = "qwen-vl-plus"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dpi: int = 144
    image_format: str = "png"
    max_pages: int = 100

    @classmethod
    def from_env(cls):
        return cls(os.getenv("QWEN_API_KEY", ""), os.getenv("QWEN_MODEL", cls.model),
                   os.getenv("QWEN_BASE_URL", cls.base_url),
                   int(os.getenv("PIPELINE_RENDER_DPI", os.getenv("QWEN_RENDER_DPI", "144"))),
                   os.getenv("PIPELINE_IMAGE_FORMAT", os.getenv("QWEN_IMAGE_FORMAT", "png")),
                   int(os.getenv("PIPELINE_MAX_PAGES", "100")))
