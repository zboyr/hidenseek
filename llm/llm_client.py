import os

from openai import OpenAI
import anthropic
import requests
from together import Together

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


class Client:
    def __init__(self):
        pass

    def get_completion(self, system: str, message: str, **generate_args):
        pass


class MistralClient(Client):
    def __init__(self, api_key, model=None):
        super().__init__()
        api_key = api_key or os.environ["MISTRAL_API_KEY"]
        from mistralai import Mistral
        from mistralai.models.systemmessage import SystemMessage
        from mistralai.models.usermessage import UserMessage
        self._SystemMessage = SystemMessage
        self._UserMessage = UserMessage
        self.client = Mistral(api_key=api_key)
        self.model = model or "mistral-medium"

    def get_completion(self, system: str, message: str, **generate_args):
        messages = []
        if system:
            messages.append(self._SystemMessage(content=system))
        messages.append(self._UserMessage(content=message))

        if "model" not in generate_args:
            generate_args["model"] = self.model

        chat_response = self.client.chat.complete(
            model=self.model,
            messages=messages,
            **{k: v for k, v in generate_args.items() if k != "model"}
        )
        return chat_response.choices[0].message.content


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
    def get_completion(self, system: str, message: str, **generate_args):
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
        return response.content[0].text


class TogetherClient(Client):
    def __init__(self, api_key, model=None):
        super().__init__()
        api_key = api_key or os.environ["TOGETHER_API_KEY"]
        self.client = Together(api_key=api_key)
        self.model = model or "meta-llama/Llama-3-8b-chat-hf"

    @retry(wait=wait_random_exponential(min=6, max=100), stop=stop_after_attempt(5))
    def get_completion(self, system: str, message: str, **generate_args):
        messages = [{"role": "user", "content": message}]
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
        return response.choices[0].message.content


class OpenRouterClient(Client):
    def __init__(self, api_key=None, model=None):
        super().__init__()
        api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        self.model = model or "meta-llama/llama-3.1-8b-instruct"

    @retry(wait=wait_random_exponential(min=6, max=100), stop=stop_after_attempt(5))
    def get_completion(self, system: str, message: str, **generate_args):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})
        if "model" not in generate_args:
            generate_args["model"] = self.model
        response = self.client.chat.completions.create(
            messages=messages,
            **generate_args
        )
        return response.choices[0].message.content


class DipperClient(Client):
    def __init__(self, endpoint_url=None, api_key=None, lex_div=20, order_div=40,
                 max_retries=10, timeout_seconds=120):
        super().__init__()
        self.endpoint_url = endpoint_url or os.environ.get("HF_DIPPER_ENDPOINT_URL")
        self.api_key = api_key or os.environ.get("HF_DIPPER_API_KEY")
        self.lex_div = lex_div
        self.order_div = order_div
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self._call_count = 0

    def get_completion(self, system: str, message: str, **generate_args):
        import time
        lex_code = int(100 - self.lex_div)
        order_code = int(100 - self.order_div)
        dipper_input = f"lexical = {lex_code}, order = {order_code} <sent> {message.strip()} </sent>"

        self._call_count += 1
        seed = generate_args.get("seed", self._call_count)
        temperature = generate_args.get("temperature", 0.75)

        payload = {
            "inputs": dipper_input,
            "parameters": {
                "do_sample": temperature > 0,
                "temperature": max(temperature, 1e-5),
                "max_new_tokens": 512,
                "seed": seed,
                "return_full_text": False,
            },
            "options": {
                "wait_for_model": True,
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        attempt = 0
        while True:
            attempt += 1
            try:
                response = requests.post(
                    self.endpoint_url, headers=headers, json=payload,
                    timeout=self.timeout_seconds,
                )
                if response.status_code >= 500:
                    raise requests.HTTPError(
                        f"Server error {response.status_code}: {response.text}",
                        response=response,
                    )
                response.raise_for_status()
                body = response.json()
                if isinstance(body, list) and body:
                    return body[0]["generated_text"]
                return body["generated_text"]
            except Exception:
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(8.0, float(2 ** attempt)))


class HumanParaphraserClient(Client):
    """Interactive human paraphraser. Embedding logged externally."""

    def __init__(self, **kwargs):
        super().__init__()

    def get_completion(self, system: str, message: str, **generate_args):
        print(f"\n{'='*60}")
        print("  HUMAN PARAPHRASER")
        print(f"{'='*60}")
        print(f"\nPlease paraphrase the following text:\n")
        print(f"  {message}\n")

        while True:
            paraphrase = input("> ").strip()
            if paraphrase:
                return paraphrase
            print("Empty input. Please enter your paraphrase.")


class PreloadedParaphraseClient(Client):
    """Returns pre-existing human paraphrases from a loaded dataset (e.g. PAR3).

    The paraphrase_map is injected after construction by AdversarialEvaluation.
    """

    def __init__(self):
        super().__init__()
        self.paraphrase_map = {}  # {source_text: [para1, para2, ...]}

    def set_paraphrase_map(self, paraphrase_map: dict):
        self.paraphrase_map = paraphrase_map

    def get_completion(self, system: str, message: str, **generate_args):
        if message in self.paraphrase_map and self.paraphrase_map[message]:
            return self.paraphrase_map[message].pop(0)
        raise ValueError(f"No pre-loaded paraphrase for: {message[:80]}...")


def client_from_args(client_str: str, **client_args):
    api_key = client_args.get("api_key")
    model = client_args.get("model")

    if client_str == "mistral":
        return MistralClient(api_key=api_key, model=model)

    elif client_str == "anthropic":
        return AnthropicClient(api_key=api_key, model=model)

    elif client_str == "together":
        return TogetherClient(api_key=api_key, model=model)

    elif client_str == "openrouter":
        return OpenRouterClient(api_key=api_key, model=model)

    elif client_str == "dipper":
        return DipperClient(
            endpoint_url=client_args.get("endpoint_url"),
            api_key=api_key,
            lex_div=client_args.get("lex_div", 20),
            order_div=client_args.get("order_div", 40),
        )

    elif client_str == "human":
        return HumanParaphraserClient(
            similarity_threshold=client_args.get("similarity_threshold", 0.75),
            embedding_model=client_args.get("embedding_model", "gemini-embedding-001"),
        )

    elif client_str == "preloaded":
        return PreloadedParaphraseClient()

    else:
        raise ValueError(
            f"supported choices are ['mistral', 'anthropic', 'together', 'openrouter', 'dipper', 'human', 'preloaded']. Got {client_str}"
        )

