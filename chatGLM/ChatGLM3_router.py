import json
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import List, Optional

class ChatGLM(LLM):
    max_token: int = 8192
    do_sample: bool = False
    temperature: float = 0.8
    top_p = 0.8
    tokenizer: object = None
    model: object = None
    history: List = []
    tool_names: List = []
    has_search: bool = False

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM3"

    def load_model(self, model_name_or_path=None):
        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path, config=model_config, trust_remote_code=True
        ).half().cuda()


    def _call(self, prompt: str, history: List = [], stop: Optional[List[str]] = ["<|user|>"]):
        return ""

    