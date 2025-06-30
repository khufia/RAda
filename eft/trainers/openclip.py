import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from fvcore.nn import FlopCountAnalysis 
from torchvision import transforms
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import open_clip



def norm1(tensor):    
    # Get the minimum and maximum along the last two dimensions    
    tensor_min = tensor.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]    
    tensor_max = tensor.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]        
    # Apply min-max normalization to [0, 1]    
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)  # Avoid division by zero        
    # Scale to [-1, 1]    
    normalized_tensor = (normalized_tensor * 2 - 1)        
    return normalized_tensor

def normbymax(tensor):    
    tensor_max = tensor.max(dim=-1, keepdim=True)[0]    
    return tensor / tensor_max


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class SimpleMultiheadAttention(nn.Module):    
    def __init__(self, embed_dim, num_heads, t, dropout=0., batch_first=True, bias=True):
        super(SimpleMultiheadAttention, self).__init__()        
        self.embed_dim = embed_dim        
        self.num_heads = num_heads        
        self.dropout = nn.Dropout(dropout)        
        self.batch_first = batch_first                
        # Ensure embedding dimension is divisible by number of heads        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."                
        self.head_dim = embed_dim // num_heads
        self.temprature = t
        # Define the linear projections for Query, Key, and Value        
        self.q_proj1 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj2 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj3 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        nn.init.constant_(self.out_proj.weight, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

        
    def forward(self, query1, query2, query3, key, value, attn_mask=None):
        # Project query, key, and value        
        q1 = self.q_proj1(query1)  # Shape: (B, T, E)
        q2 = self.q_proj2(query2)  # Shape: (B, T, E)
        q3 = self.q_proj3(query3)  # Shape: (B, T, E)
        k = self.k_proj(key)  # Shape: (B, S, E)
        v = self.v_proj(value)  # Shape: (B, S, E)
        # Reshape for multihead attention (split across the number of heads)        
        B, T, E = q1.shape
        _, S, _ = k.shape        
        num_heads = self.num_heads        
        head_dim = self.head_dim                
        # Reshape query, key, and value into (B, num_heads, seq_len, head_dim)        
        q1 = q1.view(B, T, num_heads, head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        q2 = q2.view(B, T, num_heads, head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        q3 = q3.view(B, T, num_heads, head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.view(B, S, num_heads, head_dim).transpose(1, 2)  # (B, num_heads, S, head_dim)
        v = v.view(B, S, num_heads, head_dim).transpose(1, 2)  # (B, num_heads, S, head_dim)                
        # Compute scaled dot-product attention        
        attn_weights1 = torch.matmul(q1, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights2 = torch.matmul(q2, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights3 = torch.matmul(q3, k.transpose(-2, -1)) / (head_dim ** 0.5)
        if attn_mask is not None:
            attn_weights1 += attn_mask
            attn_weights2 += attn_mask
            attn_weights3 += attn_mask
        attn_weights1 = F.softmax(attn_weights1.view(B, num_heads, -1) /
                                 self.temprature, dim=-1).view(B, num_heads, T, T) * T
        attn_weights2 = F.softmax(attn_weights2.view(B, num_heads, -1) /
                                 self.temprature, dim=-1).view(B, num_heads, T, T) * T
        attn_weights3 = F.softmax(attn_weights3.view(B, num_heads, -1) /
                                  self.temprature, dim=-1).view(B, num_heads, T, T) * T
        attn_weights1 = self.dropout(attn_weights1)  # Apply dropout
        attn_weights2 = self.dropout(attn_weights2)  # Apply dropout
        attn_weights3 = self.dropout(attn_weights3)  # Apply dropout

        # (B, num_heads, T, head_dim)
        attn_output = torch.matmul((attn_weights1 + attn_weights2 + attn_weights3)/3, v)   # {I, T, R}
        # attn_output = torch.matmul(attn_weights1, v)  # I
        # attn_output = torch.matmul(attn_weights2, v)  # (B, num_heads, T, head_dim)      # T

        attn_output = attn_output.transpose(1, 2).reshape(B, T, E)  # (B, T, E)
        output = self.out_proj(attn_output)
        return output, attn_weights1
    


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class OpenClip(nn.Module):    
    def __init__(self, cfg, classnames):
        super().__init__()
        # get classnames and features
        self.prompt_prefix = cfg.TRAINER.QKMASK.CTX_INIT        
        print('Fixed Prompts Query-Key Design')
        print(f'Initial context: "{self.prompt_prefix}"')
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.QKMASK.N_CTX}")
        print(f"Using fixed hand crafted prompts")

        classnames = [name.replace("_", " ") for name in classnames]
        self.prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        self.model, self.preprocess = open_clip.create_model_from_pretrained(
            'ViT-B-16',                    # model architecture
            pretrained='laion2b_s34b_b88k' # tag from HF
        )
        self.tokenizer = open_clip.get_tokenizer('ViT-B-16')

        self.logit_scale = self.model.logit_scale
        self.temparature = cfg.TEMP
        self.ALPHA = cfg.ALPHA
        self.criterion = nn.MSELoss()
        self.image_encoder = self.model.visual
        self.text_encoder = self.model.transformer
        self.ln_final = self.model.ln_final
        self.init_projections(cfg)


    def init_projections(self, cfg):
        dim = 512
        self.mha = SimpleMultiheadAttention(t=self.temparature, embed_dim=dim, num_heads=8)


    def forward(self, image, label=None, training=False):
        image = denormalize(image)
        to_pil = transforms.ToPILImage()
        images = [to_pil(img) for img in image]        
        images = [self.preprocess(img).unsqueeze(0) for img in images]
        images = torch.stack(images, dim=0).to('cuda')
        text = self.tokenizer(self.prompts).to('cuda')
        self.image_encoder.eval()
        self.text_encoder.eval()
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        image_features = image_features.unsqueeze(1)
        text_features = text_features.unsqueeze(0)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        IT = image_features * text_features
        B, K, D = IT.shape
        scale = logit_scale.data
        q1, q2 = image_features.repeat(1, K, 1), text_features.repeat(B, 1, 1)
                                       # (q1, q2, q3, key, value)     

        mask, attn_weights = self.mha(scale.sqrt()*q1, scale.sqrt()*q2, scale*IT, scale * IT, scale * IT)

        mask_final = torch.relu(torch.ones_like(mask) + mask)
        self.mask_weights = mask_final.clone().detach()
        masked_features = mask_final * IT

        self.RM = IT.clone().detach()
        self.ARM = masked_features.clone().detach()
       
        logits = torch.sum(masked_features, dim=-1)
        logits = logit_scale * logits

        if training:
             # Compute Contrastive loss
            loss1 = nn.CrossEntropyLoss()(logits, label)           
            loss2 = self.criterion(mask_final, torch.ones_like(mask_final))
            return (logits, loss1, loss2)
        return logits


@TRAINER_REGISTRY.register()
class OPENCLIP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.QKMASK.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")

        print("Building custom BLIP")
        self.model = OpenClip(cfg, classnames)

        for name, param in self.model.named_parameters():
            # if "projq" not in name and "projk" not in name and "projv" not in name:
            if "mha" not in name and "adapter" not in name:
            # if "mha" not in name:
                param.requires_grad_(False)

        self.model.to(self.device)
        # NOTE: only give projection weights to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        print(f"OPtim: {self.optim}")
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("QKMASK", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.QKMASK.PREC == "amp" else None

        for name, param in self.model.named_parameters():
            # if "projq" not in name and "projk" not in name and "projv" not in name:
            if "mha" not in name and "adapter" not in name:
            # if "mha" not in name:
                param.requires_grad_(False)
        
        # set image and text encoder to eval mode (frozen)
        self.model.image_encoder.eval()
        self.model.text_encoder.eval()
        self.model.ln_final.eval()
        
        # Double check
        enabled = set()
        mask_params_name = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        self.model.to(self.device)



    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.QKMASK.PREC

        # Training step
        if prec == "amp":
            with autocast():
                loss = model(image, label, training=True)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            output, loss1, loss2 = model(image, label, training=True)
            loss = loss1 + self.cfg.ALPHA*loss2
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item(), "loss1": loss1.item(),
                         "loss2": loss2.item(), 
                        "acc": compute_accuracy(output, label)[0].item()             
                        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


def denormalize(tensor):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=tensor.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=tensor.device).view(1, 3, 1, 1)

    # Denormalize
    tensor = tensor * std + mean
    
    # Clamp to [0, 1] range
    return tensor.clamp(0, 1)