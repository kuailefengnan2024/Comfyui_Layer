# comfyui_vision_plugin.py

# ==================================================================================
# I. 说明 (INSTRUCTIONS)
# ==================================================================================
# 1. 将此文件放入 ComfyUI 安装目录下的 `ComfyUI/custom_nodes/` 文件夹中。
# 2. 重启 ComfyUI。
# 3. 在 ComfyUI 中，你可以通过搜索 "Vision API Node" 来找到这个新节点。
#
# 依赖 (Dependencies):
# 请确保你的 ComfyUI 环境中安装了以下 Python 包。
# 你可以通过在 ComfyUI 的 Python 环境中运行 pip install来安装它们。
#
# pip install openai httpx
# ==================================================================================

import os
import sys
import base64
import asyncio
import mimetypes
import uuid
import torch
import numpy as np
import httpx
from PIL import Image
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any

# ComfyUI aiohttp atexit hook
import atexit
# For ComfyUI, we need to be careful with asyncio event loops.
# ComfyUI runs its own loop. We'll use a helper to run our async code.
def run_async_in_sync(coro):
    """
    Runs an async coroutine in a synchronous context, creating a new event loop
    if one isn't running in the current thread.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # If a loop is already running, we can't create a new one.
        # We need to run the coroutine in a separate thread with its own loop.
        # This is a common pattern for integrating async code into sync frameworks.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)

# Helper for ComfyUI temp files
try:
    import folder_paths
except ImportError:
    # A fallback for running the script outside ComfyUI, though it's not the primary use case.
    class MockFolderPaths:
        def get_temp_directory(self):
            return "temp"
        def get_output_directory(self):
            return "output"
    folder_paths = MockFolderPaths()
    print("Warning: `folder_paths` not found. Using mock paths. This script is intended to be run as a ComfyUI custom node.")


# ==================================================================================
# II. 从项目中复制并修改的API逻辑 (ADAPTED API LOGIC FROM PROJECT)
# ==================================================================================

# ----------------------------------------------------------------------------------
# 1. 抽象基类 (Abstract Base Class) - from api/base.py
# ----------------------------------------------------------------------------------
class BaseVisionProvider(ABC):
    """
    内容审核与视觉理解提供商的抽象基类。
    """
    @abstractmethod
    async def call_api(self, prompt_text: str, **kwargs) -> Tuple[str | None, str | None]:
        """
        调用视觉API的核心方法。
        Args:
            prompt_text (str): 发送给模型的用户文本提示。
            **kwargs: 必须包含 'image_path' (str) 来指定图片。
        Returns:
            一个元组 (content, error_message)。
        """
        raise NotImplementedError

    async def close(self):
        """可选的关闭或清理资源的方法。"""
        pass

# ----------------------------------------------------------------------------------
# 2. Gemini-2.5-Pro 模型实现 (Gemini-2.5-Pro Implementation) - from api/vision/gemini_2_5_pro.py
# ----------------------------------------------------------------------------------
from openai import AsyncAzureOpenAI

async def _get_image_uri(image_path_or_url: str) -> str | None:
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
        return await _fetch_and_encode_image_from_url(image_path_or_url)
    else:
        return _encode_local_image(image_path_or_url)

def _encode_local_image(image_path: str) -> str | None:
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"
        encoded_string = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"[VisionAPIPluginNode] Error encoding local image {image_path}: {e}")
        return None

async def _fetch_and_encode_image_from_url(image_url: str) -> str | None:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, follow_redirects=True)
            response.raise_for_status()
        image_bytes = response.content
        mime_type = response.headers.get("content-type")
        if not mime_type or not mime_type.startswith("image/"):
            mime_type, _ = mimetypes.guess_type(image_url)
            if not mime_type:
                mime_type = "image/jpeg"
        encoded_string = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"[VisionAPIPluginNode] Error fetching image from URL {image_url}: {e}")
        return None

class Gemini25ProVisionProvider(BaseVisionProvider):
    def __init__(self, api_key: str, azure_endpoint: str, api_version: str, deployment_name: str, **kwargs):
        if not all([api_key, azure_endpoint, api_version, deployment_name]):
            raise ValueError("For Gemini Pro Vision, API key, endpoint, API version, and deployment name are required.")
        
        self.deployment_name = deployment_name
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        print(f"[VisionAPIPluginNode] Gemini25ProVisionProvider initialized for deployment: {self.deployment_name}")

    async def call_api(self, prompt_text: str, **kwargs) -> Tuple[str | None, str | None]:
        image_path = kwargs.get("image_path")
        if not image_path:
            return None, "Image path is required for Vision API call."
        
        encoded_image_uri = await _get_image_uri(image_path)
        if not encoded_image_uri:
            return None, f"Could not process image from source: {image_path}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": encoded_image_uri}},
                ],
            }
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 2048)
            )
            content = response.choices[0].message.content
            return content, None
        except Exception as e:
            print(f"[VisionAPIPluginNode] Error calling Gemini 2.5 Pro Vision API: {e}")
            return None, str(e)

    async def close(self):
        if self.client:
            await self.client.close()
            print("[VisionAPIPluginNode] Gemini25ProVisionProvider client closed.")

# ----------------------------------------------------------------------------------
# 3. 豆包模型实现 (Doubao Model Implementation) - from api/vision/doubao_seed_vision.py
# ----------------------------------------------------------------------------------
from openai import AsyncOpenAI

def encode_image_to_base64_doubao(image_path: str) -> str:
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except IOError as e:
        print(f"[VisionAPIPluginNode] Could not read image file {image_path}: {e}")
        raise

class DoubaoSeedVisionProvider(BaseVisionProvider):
    def __init__(self, model: str, api_key: str, base_url: str, **kwargs):
        if not all([api_key, base_url, model]):
            raise ValueError("For Doubao Vision, API key, base URL, and model name are required.")
        
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        print(f"[VisionAPIPluginNode] DoubaoSeedVisionProvider initialized for model: {self.model}")

    async def call_api(self, prompt_text: str, **kwargs) -> Tuple[str | None, str | None]:
        image_path = kwargs.get("image_path")
        if not image_path:
            return None, "Image path is required for Doubao vision call."

        try:
            # Note: Doubao's async call doesn't need to be in a separate thread like in the original code
            # as we are already in an async context managed by run_async_in_sync.
            base64_image = encode_image_to_base64_doubao(image_path)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            
            api_kwargs = kwargs.copy()
            api_kwargs.pop("image_path", None)

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **api_kwargs
            )
            
            if not response.choices:
                return None, "API call successful, but returned no choices."

            content = response.choices[0].message.content
            return content, None
        except Exception as e:
            print(f"[VisionAPIPluginNode] Error calling Doubao Vision API: {e}")
            return None, str(e)

    async def close(self):
        if self.client:
            await self.client.close()
            print("[VisionAPIPluginNode] DoubaoSeedVisionProvider client closed.")


# ==================================================================================
# III. COMFYUI 节点定义 (COMFYUI NODE DEFINITION)
# ==================================================================================
class VisionAPIPluginNode:
    def __init__(self):
        self.temp_dir = folder_paths.get_temp_directory()
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
            
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Describe this image in detail."}),
                "model_selection": (["gemini-pro-vision", "doubao-seed-vision"],),
                
                # Gemini (Azure) specific inputs
                "api_key_gemini": ("STRING", {"multiline": False, "default": "YOUR_AZURE_OPENAI_KEY"}),
                "azure_endpoint_gemini": ("STRING", {"multiline": False, "default": "https://your-endpoint.openai.azure.com/"}),
                "api_version_gemini": ("STRING", {"multiline": False, "default": "2024-05-01-preview"}),
                "deployment_name_gemini": ("STRING", {"multiline": False, "default": "gpt-4o"}),

                # Doubao (VolcEngine) specific inputs
                "api_key_doubao": ("STRING", {"multiline": False, "default": "YOUR_ARK_API_KEY"}),
                "base_url_doubao": ("STRING", {"multiline": False, "default": "https://ark.cn-beijing.volces.com/api/v3"}),
                "model_doubao": ("STRING", {"multiline": False, "default": "ep-20240621110657-p67m9"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "Vision"

    def tensor_to_pil(self, tensor):
        """Converts a ComfyUI IMAGE tensor (Batch, Height, Width, Channel) to a PIL Image."""
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)  # Remove batch dimension if present
        
        # The tensor from ComfyUI is already in HWC format (Height, Width, Channel).
        # No permutation is needed.
        image_np = tensor.cpu().numpy()
        
        # Denormalize from 0-1 to 0-255 and convert to uint8
        image_np = (image_np * 255).astype(np.uint8)
        return Image.fromarray(image_np)

    def run(self, image, prompt, model_selection, 
            api_key_gemini, azure_endpoint_gemini, api_version_gemini, deployment_name_gemini,
            api_key_doubao, base_url_doubao, model_doubao):
            
        pil_image = self.tensor_to_pil(image)
        
        # Save temp image
        filename = f"vision_api_temp_{uuid.uuid4()}.png"
        filepath = os.path.join(self.temp_dir, filename)
        pil_image.save(filepath)

        client = None
        content = None
        error = "An unknown error occurred."

        try:
            if model_selection == "gemini-pro-vision":
                if not api_key_gemini or "YOUR_AZURE_OPENAI_KEY" in api_key_gemini:
                     return ("Error: Gemini API Key is not set.",)
                try:
                    client = Gemini25ProVisionProvider(
                        api_key=api_key_gemini,
                        azure_endpoint=azure_endpoint_gemini,
                        api_version=api_version_gemini,
                        deployment_name=deployment_name_gemini
                    )
                except ValueError as e:
                    return (f"Error initializing Gemini client: {e}",)

            elif model_selection == "doubao-seed-vision":
                if not api_key_doubao or "YOUR_ARK_API_KEY" in api_key_doubao:
                    return ("Error: Doubao API Key is not set.",)
                try:
                    client = DoubaoSeedVisionProvider(
                        api_key=api_key_doubao,
                        base_url=base_url_doubao,
                        model=model_doubao
                    )
                except ValueError as e:
                    return (f"Error initializing Doubao client: {e}",)
            
            if client:
                # Run the async call
                content, error = run_async_in_sync(client.call_api(prompt, image_path=filepath))
                # Clean up client resources
                run_async_in_sync(client.close())

        finally:
            # Clean up temp file
            if os.path.exists(filepath):
                os.remove(filepath)

        if error:
            return (f"API Error: {error}",)
        
        return (content or "No content returned.",)

