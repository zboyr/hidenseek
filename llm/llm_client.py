import os
from typing import Any

try:
    from mistralai.client import MistralClient as _LegacyMistralClient
except Exception:
    _LegacyMistralClient = None

try:
    from mistralai import Mistral as _NewMistralClient
except Exception:
    _NewMistralClient = None

# from openai import OpenAI
import anthropic
from together import Together

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


class Client:
    def __init__(self):
        pass

    def get_completion(self, system: str, message: str, **generate_args) -> str:
        raise NotImplementedError


class MistralClient(Client):
    def __init__(self, api_key, model=None):
        super().__init__()
        api_key = api_key or os.environ["MISTRAL_API_KEY"]
        self.api_mode = None

        if _LegacyMistralClient is not None:
            legacy_client: Any = _LegacyMistralClient(api_key=api_key)
            if hasattr(legacy_client, "chat"):
                self.client = legacy_client
                self.api_mode = "legacy"
            elif _NewMistralClient is not None:
                self.client = _NewMistralClient(api_key=api_key)
                self.api_mode = "new"
            else:
                raise ImportError("No compatible Mistral client API found.")
        elif _NewMistralClient is not None:
            self.client = _NewMistralClient(api_key=api_key)
            self.api_mode = "new"
        else:
            raise ImportError("mistralai package is unavailable or incompatible.")

        self.model = model or "mistral-medium"

    @staticmethod
    def _normalize_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text:
                        parts.append(str(text))
                else:
                    text = getattr(item, "text", None) or getattr(item, "content", None)
                    if text:
                        parts.append(str(text))
            return "".join(parts)
        return str(content)

    def get_completion(self, system: str, message: str, **generate_args) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        if "model" not in generate_args:
            generate_args["model"] = self.model

        if "temperature" not in generate_args:
            generate_args["temperature"] = 0.3

        if self.api_mode == "legacy":
            chat_response = self.client.chat(
                messages=messages,
                **generate_args,
            )
            return self._normalize_content(chat_response.choices[0].message.content)

        request_args: dict[str, Any] = {
            "model": generate_args["model"],
            "messages": messages,
            "temperature": generate_args["temperature"],
        }

        passthrough_args = ["top_p", "max_tokens", "stop", "presence_penalty", "frequency_penalty"]
        for arg_name in passthrough_args:
            if arg_name in generate_args:
                request_args[arg_name] = generate_args[arg_name]

        if "seed" in generate_args:
            request_args["random_seed"] = generate_args["seed"]

        chat_response = self.client.chat.complete(**request_args)
        return self._normalize_content(chat_response.choices[0].message.content)


# class OpenAIClient(Client):
#     def __init__(self, api_key, model=None, url=None):
#         super().__init__()
#         api_key = api_key or os.environ["OPENAI_API_KEY"]
#         self.client = OpenAI(api_key=api_key, base_url=url)
#         self.model = model or "gpt-4"
#
#     @retry(wait=wait_random_exponential(min=6, max=100), stop=stop_after_attempt(5))
#     def get_completion(self, system: str, message: str, **generate_args):
#         messages = []
#         if system:
#             messages.append({"role": "system", "content": system})
#         messages.append({"role": "user", "content":  message})
#         if "model" not in generate_args:
#             generate_args["model"] = self.model
#         chat_response = self.client.chat.completions.create(
#             messages=messages,
#             **generate_args
#         )
#         return chat_response.choices[0].message.content


class AnthropicClient(Client):
    def __init__(self, api_key, model=None):
        super().__init__()
        api_key = api_key or os.environ["ANTHROPIC_API_KEY"]
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or "claude-3-opus-20240229"

    @retry(wait=wait_random_exponential(min=6, max=100), stop=stop_after_attempt(5))
    def get_completion(self, system: str, message: str, **generate_args) -> str:
        messages = []
        messages.append({"role": "user", "content": message})

        model = self.model
        if "model" in generate_args:
            model = generate_args["model"]

        generate_args["model"] = model

        if "temperature" not in generate_args:
            generate_args["temperature"] = 0.3

        response = self.client.messages.create(
            messages=messages,
            max_tokens=4096,
            model = model,
            system = system,
            temperature=generate_args["temperature"]
        )
        if not response.content:
            return ""
        block = response.content[0]
        text = getattr(block, "text", None)
        if text is not None:
            return str(text)
        fallback = getattr(block, "content", None)
        if fallback is not None:
            return str(fallback)
        return str(block)


class TogetherClient(Client):
    def __init__(self, api_key, model=None):
        super().__init__()
        api_key = api_key or os.environ["TOGETHER_API_KEY"]
        self.client = Together(api_key=api_key)
        self.model = model or "meta-llama/Llama-3-8b-chat-hf"

    @retry(wait=wait_random_exponential(min=6, max=100), stop=stop_after_attempt(5))
    def get_completion(self, system: str, message: str, **generate_args) -> str:
        messages: list[Any] = [{"role": "user", "content": message}]
        if system:
            messages.insert(0, {"role": "system", "content": system})

        if "model" not in generate_args:
            generate_args["model"] = self.model
        
        if "temperature" not in generate_args:
            generate_args["temperature"] = 1.0

        response = self.client.chat.completions.create(
            model=generate_args["model"],
            messages=messages,
            temperature=generate_args["temperature"]
        )
        if not response.choices:
            return ""
        content = response.choices[0].message.content
        if content is None:
            return ""
        return str(content)


def client_from_args(client_str: str, **client_args):
    api_key = client_args.get("api_key")
    model = client_args.get("model")

    if client_str == "mistral":
        return MistralClient(api_key=api_key, model=model)

    # elif client_str == "openai":
    #     return OpenAIClient(api_key=api_key, model=model)

    elif client_str == "anthropic":
        return AnthropicClient(api_key=api_key, model=model)

    elif client_str == "together":
        return TogetherClient(api_key=api_key, model=model)

    else:
        raise ValueError(f"supported choices are ['mistral', 'openai', 'anthropic', 'together']. Got {client_str}")
