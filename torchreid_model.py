"""
TorchReID模型封装 (最终修复版)
- 彻底依赖 torchreid 库进行模型创建，移除自定义实现。
- 确保在加载自定义权重时，传入正确的 num_classes。
"""
import torch
import torchvision.transforms as T
import cv2

# 确保 torchreid 库已安装
try:
    import torchreid
except ImportError:
    raise ImportError("错误: 未找到 torchreid 库。请运行 'pip install torchreid'。")

class RealReIDModel:
    """ReID模型封装类 (简化且健壮)"""
    
    def __init__(self, model_path, model_name='osnet_ain_x1_0', num_classes=9, device='cuda'):
        """
        初始化ReID模型
        Args:
            model_path (str): 你训练好的 .pth 模型权重文件路径。
            model_name (str): 模型架构名称，必须与训练时一致。
            num_classes (int): 训练时使用的训练集ID数量，必须一致。
            device (str): 'cuda' 或 'cpu'。
        """
        self.device = torch.device(device)
        self.model_name = model_name
        
        # 1. 创建一个与你训练时结构完全相同的空模型
        #    这是加载自定义权重的关键
        print(f"  - [ReID] 创建模型 '{model_name}' (为 {num_classes} 个类别)...")
        self.model = torchreid.models.build_model(
            name=self.model_name,
            num_classes=num_classes,
            pretrained=False  # 我们要加载自己的权重，所以这里设为False
        )
        
        # 2. 加载你训练好的权重
        print(f"  - [ReID] 从 '{model_path}' 加载Fine-tuned权重...")
        state_dict = torch.load(model_path, map_location=self.device)
        
        # 处理可能的'state_dict'键
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        self.model.load_state_dict(state_dict)
        print("  - [ReID] 权重加载成功！")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 3. 定义图像预处理变换
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def extract_feature(self, image_bgr):
        """
        提取单张图像的ReID特征
        Args:
            image_bgr: BGR格式的NumPy图像数组。
        Returns:
            NumPy数组形式的特征向量，或在失败时返回None。
        """
        if image_bgr is None or image_bgr.size == 0:
            return None
        
        try:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            # torchreid模型在eval模式下默认只输出特征
            features = self.model(input_tensor)
            
            # L2归一化
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            return features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None