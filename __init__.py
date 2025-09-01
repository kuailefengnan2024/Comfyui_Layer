import torch
import numpy as np
from PIL import Image, ImageDraw
import comfy.model_management

class ImageLayerComposite:
    """
    图层叠加节点 - 将图1叠加到图2上，支持调整大小和位置
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_base": ("IMAGE",),  # 底层图片（图2）
                "image_overlay": ("IMAGE",),  # 上层图片（图1）
                "overlay_width": ("INT", {
                    "default": 100, 
                    "min": 1, 
                    "max": 2048, 
                    "step": 1,
                    "display": "number"
                }),
                "overlay_height": ("INT", {
                    "default": 100, 
                    "min": 1, 
                    "max": 2048, 
                    "step": 1,
                    "display": "number"
                }),
                "x_position": ("INT", {
                    "default": 0, 
                    "min": -2048, 
                    "max": 2048, 
                    "step": 1,
                    "display": "number"
                }),
                "y_position": ("INT", {
                    "default": 0, 
                    "min": -2048, 
                    "max": 2048, 
                    "step": 1,
                    "display": "number"
                }),
                "opacity": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "display": "number"
                }),
            },
            "optional": {
                "keep_aspect_ratio": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composite_image",)
    FUNCTION = "composite_layers"
    CATEGORY = "image/composite"
    
    def composite_layers(self, image_base, image_overlay, overlay_width, overlay_height, 
                        x_position, y_position, opacity, keep_aspect_ratio=True):
        """
        执行图层叠加操作
        """
        # 转换tensor到PIL Image
        base_pil = self.tensor_to_pil(image_base[0])
        overlay_pil = self.tensor_to_pil(image_overlay[0])
        
        # 获取底图尺寸
        base_width, base_height = base_pil.size
        
        # 调整上层图片大小
        if keep_aspect_ratio:
            # 保持纵横比
            overlay_ratio = overlay_pil.width / overlay_pil.height
            if overlay_width / overlay_height > overlay_ratio:
                # 以高度为准
                new_height = overlay_height
                new_width = int(overlay_height * overlay_ratio)
            else:
                # 以宽度为准
                new_width = overlay_width
                new_height = int(overlay_width / overlay_ratio)
        else:
            # 不保持纵横比，直接拉伸
            new_width = overlay_width
            new_height = overlay_height
        
        # 调整overlay图片大小
        overlay_resized = overlay_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 如果不是RGBA模式，转换为RGBA以支持透明度
        if overlay_resized.mode != 'RGBA':
            overlay_resized = overlay_resized.convert('RGBA')
        
        # 调整透明度
        if opacity < 1.0:
            # 创建一个alpha通道
            overlay_array = np.array(overlay_resized)
            overlay_array[:, :, 3] = (overlay_array[:, :, 3] * opacity).astype(np.uint8)
            overlay_resized = Image.fromarray(overlay_array, 'RGBA')
        
        # 确保底图是RGB模式
        if base_pil.mode != 'RGB':
            base_pil = base_pil.convert('RGB')
        
        # 创建最终合成图像
        result = base_pil.copy()
        
        # 计算实际粘贴位置（处理负坐标和边界）
        paste_x = max(0, x_position)
        paste_y = max(0, y_position)
        
        # 计算overlay的裁剪区域（如果部分超出边界）
        crop_left = max(0, -x_position)
        crop_top = max(0, -y_position)
        crop_right = min(new_width, base_width - x_position + crop_left)
        crop_bottom = min(new_height, base_height - y_position + crop_top)
        
        # 裁剪overlay（如果需要）
        if crop_left > 0 or crop_top > 0 or crop_right < new_width or crop_bottom < new_height:
            overlay_resized = overlay_resized.crop((crop_left, crop_top, crop_right, crop_bottom))
        
        # 粘贴overlay到底图上
        try:
            result.paste(overlay_resized, (paste_x, paste_y), overlay_resized)
        except:
            # 如果粘贴失败，尝试转换为RGB模式
            overlay_rgb = Image.new('RGB', overlay_resized.size, (255, 255, 255))
            overlay_rgb.paste(overlay_resized, mask=overlay_resized.split()[-1] if overlay_resized.mode == 'RGBA' else None)
            result.paste(overlay_rgb, (paste_x, paste_y))
        
        # 转换回tensor格式
        result_tensor = self.pil_to_tensor(result)
        
        return (result_tensor,)
    
    def tensor_to_pil(self, tensor):
        """将ComfyUI的tensor格式转换为PIL Image"""
        # tensor格式: [height, width, channels]
        if len(tensor.shape) == 3:
            # 转换为numpy数组
            array = tensor.cpu().numpy()
            # 确保值在0-255范围内
            array = np.clip(array * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(array)
        else:
            raise ValueError("输入tensor格式不正确")
    
    def pil_to_tensor(self, pil_image):
        """将PIL Image转换为ComfyUI的tensor格式"""
        # 转换为numpy数组
        array = np.array(pil_image).astype(np.float32) / 255.0
        # 转换为tensor并添加batch维度
        tensor = torch.from_numpy(array)[None,]
        return tensor

# 节点映射
NODE_CLASS_MAPPINGS = {
    "ImageLayerComposite": ImageLayerComposite
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageLayerComposite": "图层叠加"
}