import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip, tokenize, load
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

DOWNLOAD_ROOT_v2 = '/l/users/zhiqiang.shen/ghazi/DATA/CLIP'

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME

    model, _, _ = load(backbone_name, device='cpu', download_root=DOWNLOAD_ROOT_v2)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts



# try 6 layers with 3 residuals
class ProjBlock(nn.Module):
    def __init__(self, dim, n_layers=4, proj_type='key'):
        super(ProjBlock, self).__init__()
        self.proj_type = proj_type
        layers = []
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(dim, dim))
            if proj_type != 'value': 
                layers.append(nn.ReLU())
        layers.append(nn.Linear(dim, dim))
        self.block = nn.Sequential(*layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)


    # out = layer(x) + x
    def forward(self, x):
        out = self.block(x)
        return out 
    

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
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj2 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj3 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        nn.init.constant_(self.out_proj.weight, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

        
    def forward(self, query1, query2, key, value, attn_mask=None):
        # Project query, key, and value        
        q = self.q_proj(query1)  # Shape: (B, T, E)
        q2 = self.q_proj2(query2)  # Shape: (B, T, E)
        q3 = self.q_proj3(key)  # Shape: (B, T, E)
        k = self.k_proj(key)  # Shape: (B, S, E)
        v = self.v_proj(value)  # Shape: (B, S, E)
        # Reshape for multihead attention (split across the number of heads)        
        B, T, E = q.shape
        _, S, _ = k.shape        
        num_heads = self.num_heads        
        head_dim = self.head_dim                
        # Reshape query, key, and value into (B, num_heads, seq_len, head_dim)        
        q = q.view(B, T, num_heads, head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        q2 = q2.view(B, T, num_heads, head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        q3 = q3.view(B, T, num_heads, head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.view(B, S, num_heads, head_dim).transpose(1, 2)  # (B, num_heads, S, head_dim)
        v = v.view(B, S, num_heads, head_dim).transpose(1, 2)  # (B, num_heads, S, head_dim)                
        # Compute scaled dot-product attention        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights2 = torch.matmul(q2, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights3 = torch.matmul(q3, k.transpose(-2, -1)) / (head_dim ** 0.5)
        if attn_mask is not None:
            attn_weights += attn_mask
            attn_weights2 += attn_mask
        attn_weights = F.softmax(attn_weights.view(B, num_heads, -1) /
                                 self.temprature, dim=-1).view(B, num_heads, T, T) * T
        attn_weights2 = F.softmax(attn_weights2.view(B, num_heads, -1) /
                                 self.temprature, dim=-1).view(B, num_heads, T, T) * T
        attn_weights3 = F.softmax(attn_weights3.view(B, num_heads, -1) /
                                  self.temprature, dim=-1).view(B, num_heads, T, T) * T
        attn_weights = self.dropout(attn_weights)  # Apply dropout
        attn_weights2 = self.dropout(attn_weights2)  # Apply dropout
        attn_weights3 = self.dropout(attn_weights3)  # Apply dropout
        attn_output = torch.matmul((attn_weights + attn_weights2 + attn_weights3)/3, v)  # (B, num_heads, T, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, T, E)  # (B, T, E)
        output = self.out_proj(attn_output)
        return output, attn_weights
       

class CustomCLIP(nn.Module):    
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        # set up the tokens for text encoder
        self.embed_dim = 512
        # get classnames and features
        print('Fixed Prompts Query-Key Design')
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.MASKCOOP.N_CTX}")
        print(f"Using fixed hand crafted prompts")

        self.image_encoder = clip_model.visual
        self.ln_final = clip_model.ln_final
        self.logit_scale = clip_model.logit_scale

        self.temparature = cfg.TEMP
        # initialize query and projection layers
        self.init_projections(cfg)
        self.criterion = nn.L1Loss()
        self.criterion2 = nn.MSELoss()
        self.dtype = clip_model.dtype
        self.alpha = cfg.ALPHA


    @torch.no_grad()
    def get_class_features(self, tokenized_prompts, clip_model):
        class_features = clip_model.encode_text(tokenized_prompts)
        # class_features = class_features / class_features.norm(dim=-1, keepdim=True)
        return class_features.cuda()


    def init_projections(self, cfg):
        dim = self.embed_dim
        n_layers = cfg.TRAINER.QKMASK.PROJ_LAYERS
                    
        # self.mha = nn.MultiheadAttention(dim, num_heads=8, batch_first=True, bias=True)
        self.mha = SimpleMultiheadAttention(t=self.temparature, embed_dim=dim, num_heads=8)

        self.logit_scale = nn.Parameter(self.logit_scale)


    def forward(self, image, label=None, training=False):
        with torch.no_grad():
            image_features = self.image_encoder(image)   # K, D 
            
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = image_features.unsqueeze(1)
        text_features = text_features.unsqueeze(0)

        logit_scale = self.logit_scale.exp()

        IT = image_features * text_features
        B, K ,D = IT.shape
        scale = 4*logit_scale.data
        q1, q2 = image_features.repeat(1, K, 1), text_features.repeat(B, 1, 1)
        mask, attn_weights = self.mha(scale.sqrt()*q1, scale.sqrt()*q2, scale * IT, scale * IT)
        mask_final = torch.relu(torch.ones_like(mask) + mask)
        self.mask_weights = mask_final.clone().detach()
        masked_features = mask_final * IT
        logits = torch.sum(masked_features, dim=-1)

        logits = logit_scale * logits
        if training:
            loss1 = F.cross_entropy(logits, label)
            loss2 = self.criterion2(mask_final, torch.ones_like(mask_final))
            return (loss1, loss2)

        return logits
    

    def compute_mask_weights(self, image_features, text_features):
        query = self.projq(image_features.detach())  # B, D
        key = self.projk(text_features.detach())     # K, D

        query = query.unsqueeze(1)  # B, 1, D
        key = key.unsqueeze(0)   # 1, K, D

        mask = query * key   # B, K, D
        # mask = torch.ones_like(mask) + mask

        # normalize the mask weights
        mask = torch.relu(mask)
        # mask = mask / mask.max(-1, keepdim=True)[0].detach()  # (0, 1]
        return mask
    
    
    def get_mask_weights(self):
        return self.mask_weights



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class MASKCOOP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.QKMASK.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.QKMASK.PREC == "fp32" or cfg.TRAINER.QKMASK.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        for name, param in self.model.named_parameters():
            # if "projq" not in name and "projk" not in name and "projv" not in name:
            if "mha" not in name and "logit_scale" not in name and "scaling_factor" not in name and "prompt_learner" not in name:
            # if "mha" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

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
            if "projq" in name or "projk" in name:
                mask_params_name.append(name)
        print(f"Parameters to be updated: {enabled}")


        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give projection weights to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        print(f"OPtim: {self.optim}")
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("QKMASK", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.QKMASK.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            # self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.QKMASK.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label, training=True)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss1, loss2 = model(image, label, training=True)
            loss = loss1 + self.cfg.ALPHA*loss2
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item(), "loss1": loss1.item(), "loss2": loss2.item()}

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
