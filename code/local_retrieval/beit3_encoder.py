"""
beit3_encoder.py
================
Bộ mã hóa hình ảnh Image-to-Image sử dụng mô hình BEiT-3 Large (384x384 resolution)
Checkpoint: beit3_large_patch16_384_coco_retrieval.pth
Đầu ra: Vector 1024 chiều chuẩn hóa L2 (L2-Normalized).
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

from torchscale.architecture.config import EncoderConfig
from torchscale.model.BEiT3 import BEiT3
from config import DATA_DIR

class BEiT3ForRetrieval(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.beit3 = BEiT3(config)
        self.language_head = nn.Linear(config.encoder_embed_dim, config.encoder_embed_dim, bias=False)
        self.vision_head = nn.Linear(config.encoder_embed_dim, config.encoder_embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image=None):
        if image is not None:
            # 1. Vision patch embedding
            x = self.beit3.vision_embed(image)
            # 2. Multiway Transformer Encoder
            encoder_out = self.beit3.encoder(
                None,
                token_embeddings=x,
                multiway_split_position=-1,
                return_all_hiddens=True
            )
            # 3. Lấy layer cuối cùng
            last_hidden = encoder_out["encoder_states"][-1]
            # 4. CLS token (index 0)
            cls_rep = last_hidden[:, 0, :]
            # 5. Vision projection head
            vision_rep = self.vision_head(cls_rep)
            return vision_rep
        return None

class BEiT3ImageEncoder:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(BEiT3ImageEncoder, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, ckpt_path=None, device=None):
        if self._initialized:
            return
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_path = ckpt_path or os.path.join(DATA_DIR, "beit3_weights", "beit3_large_patch16_384_coco_retrieval.pth")
        
        self.config = EncoderConfig(
            encoder_layers=24,
            encoder_embed_dim=1024,
            encoder_ffn_embed_dim=4096,
            encoder_attention_heads=16,
            multiway=True,
            vocab_size=64010,
            img_size=384,
            patch_size=16,
            in_chans=3,
            max_source_positions=1024,
            checkpoint_activations=False,
        )
        
        print(f"[BEiT3ImageEncoder] Đang khởi tạo BEiT-3 Large trên thiết bị: {self.device}...")
        self.model = BEiT3ForRetrieval(self.config)
        
        if os.path.exists(self.ckpt_path):
            state_dict = torch.load(self.ckpt_path, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[BEiT3ImageEncoder] ✅ Đã nạp checkpoint thành công: {os.path.basename(self.ckpt_path)}")
        else:
            print(f"[BEiT3ImageEncoder] ⚠️ CẢNH BÁO: Không tìm thấy checkpoint tại {self.ckpt_path}")

        self.model = self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._initialized = True

    def encode_image(self, image_input) -> np.ndarray:
        """
        Nhận vào đường dẫn file ảnh, đối tượng PIL.Image hoặc numpy array,
        trả về vector 1024 chiều float32 đã chuẩn hóa L2 (norm=1.0).
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert("RGB")
        else:
            raise ValueError(f"Định dạng ảnh không hỗ trợ: {type(image_input)}")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(image=tensor)
            feat = torch.nn.functional.normalize(feat, dim=-1).squeeze(0).cpu().numpy().astype(np.float32)

        return feat
