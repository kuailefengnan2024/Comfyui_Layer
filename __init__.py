# 导入节点类
from .comfyui_vision_plugin import VisionAPIPluginNode

# 创建节点类映射
NODE_CLASS_MAPPINGS = {
    "VisionAPIPluginNode": VisionAPIPluginNode
}

# 创建节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "VisionAPIPluginNode": "Vision API Node"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
