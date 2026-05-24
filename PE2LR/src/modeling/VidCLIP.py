import torch
import torch.nn as nn
from functools import partial
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from src.modeling.CLIP_ViP import CLIPModel, clip_loss
from src.modeling.CLIP import CLIPModel as CLIP
from src.optimization.loss import UncertaintyAwareLoss, VarianceLoss, KLdivergence
from src.module.text_proxy import TextProxy
from src.module.prob_encoder import ProbTransformer, SemTransformer, GPO
from src.prob_models.pie_model import PIENet
from src.prob_models.uncertainty_module import UncertaintyModuleImage
from src.prob_models.tensor_utils import l2_normalize, sample_gaussian_tensors
import torch.nn.functional as F

class VidCLIP(nn.Module):
    def __init__(self, args):
        super(VidCLIP, self).__init__()
        clipconfig = CLIPConfig.from_pretrained(args.clip_config) # openai/clip-vit-base-patch32
        setattr(clipconfig, "vision_additional_config", args.clip_vision_additional_config)
        self.vision_additional_config = args.clip_vision_additional_config # ViP
        if args.clip_weights: # openai/clip-vit-base-patch32
            if self.vision_additional_config.type == "ViP":
                self.clipmodel = CLIPModel.from_pretrained(args.clip_weights, config=clipconfig)
            else:
                self.clipmodel = CLIP.from_pretrained(args.clip_weights, config=clipconfig)
        else:
            if self.vision_additional_config.type == "ViP":
                self.clipmodel = CLIPModel(clipconfig)
            else:
                self.clipmodel = CLIP(clipconfig)
        
        # init logit scale from 
        logit_scale_value = self.vision_additional_config.logit_scale_init_value # 4.60
        self.clipmodel.logit_scale.data.fill_(logit_scale_value)

        self.tau = 5
        self.loss_uct = UncertaintyAwareLoss(self.tau)
        
        self.prototype = nn.parameter.Parameter(torch.zeros(8, 512), requires_grad=True)
        nn.init.xavier_uniform_(self.prototype) 
        # video prototype
        self.v_prototype = nn.parameter.Parameter(torch.zeros(8, 512), requires_grad=True)
        nn.init.xavier_uniform_(self.v_prototype) 
        # text prototype
        self.t_prototype = nn.parameter.Parameter(torch.zeros(8, 512), requires_grad=True)
        nn.init.xavier_uniform_(self.t_prototype) 
        self.vis_classifier = nn.Sequential(nn.Linear(512, 8), nn.ReLU(inplace=True))
        self.text_classifier = nn.Sequential(nn.Linear(512, 8), nn.ReLU(inplace=True))

        embed_dim = 512
        self.pie_net_vis = PIENet(1, embed_dim, embed_dim, embed_dim // 2)
        self.uncertain_net_vis = UncertaintyModuleImage(embed_dim, embed_dim, embed_dim // 2)
        self.pie_net_text = PIENet(1, embed_dim, embed_dim, embed_dim // 2)
        self.uncertain_net_text = UncertaintyModuleImage(embed_dim, embed_dim, embed_dim // 2)
        self.vib_loss = KLdivergence()
        self.factor_text = nn.Parameter(torch.rand(1), requires_grad=True)
        self.factor_vis = nn.Parameter(torch.rand(1), requires_grad=True)

        self.lambda1 = 0.0001  
        self.lambda2 = 0.05    


    def overload_logit_scale(self, overload_logit_scale):
        self.clipmodel.logit_scale.data.fill_(overload_logit_scale)

    def forward(self, is_train, step, video, text_input_ids, text_input_mask,\
                image=None, caption_ids=None, caption_masks=None):
        """
        video [B, n_clips*num_frms, C, H, W]
        text_input_ids [B, L]
        text_input_mask [B, L]
        image [B, img_num, C, H, W]
        caption_ids [B, img_num, L]
        caption_masks [B, img_num, L]
        """
        B, N, C, H, W = video.shape

        if self.vision_additional_config.type == "ViP":
            inputs = {"input_ids": text_input_ids,
                    "attention_mask": text_input_mask,
                    "pixel_values": video,
                    "return_loss": False,
                    }
            outputs = self.clipmodel(**inputs)
            results = {}
            results["text_features"] = outputs["text_embeds"]    
            results["vis_features"] = outputs["image_embeds"] # (b,dim)
            vis_feat = results["vis_features"] # (b,dim)
            text_feat = results["text_features"] # (a,dim)
            wo_norm_text = outputs["wo_norm_text"] # (a,dim)
            wo_norm_vis = outputs["wo_norm_vis"] # (a,dim)

            results['text_word_features'] = outputs["text_model_output"][0]
            results['vis_patch_features'] = outputs['vision_model_output'][0][:, :self.vision_additional_config.add_cls_num + 1, :]   
            # results['vis_proxy_features'] = outputs['vision_model_output'][0][:, :self.vision_additional_config.add_cls_num + 1, :]    
            # results['vis_patch_features'] = outputs['vision_model_output'][0][:, self.vision_additional_config.add_cls_num + 1:, :]
            # for text proxy learning
            text_word_feat = results['text_word_features']  # (b,M,dim)
            vis_patch_feat = results['vis_patch_features']  # (b,M,dim)


            prob_video = self.probabilistic_video(vis_feat, F.normalize(vis_patch_feat, p=2, dim=-1).contiguous())
            prob_video_embedding = prob_video['embedding']      # 从分布中采样m个embedding
            prob_video_logsigma = prob_video['logsigma']         # 方差
            prob_video_mean = prob_video['mean']

            prob_text = self.probabilistic_text(text_feat, F.normalize(text_word_feat, p=2, dim=-1).contiguous())
            prob_text_embedding = prob_text['embedding']       # b n 512
            prob_text_logsigma = prob_text['logsigma']   # bs 512
            prob_text_mean = prob_text['mean']       # bs 512

            t2p = torch.einsum('ad, akd->ak', F.normalize(prob_text_mean, p=2, dim=-1), F.normalize(prob_text_embedding, p=2, dim=-1))
            v2p = torch.einsum('bd, bkd->bk', F.normalize(prob_video_mean, p=2, dim=-1), F.normalize(prob_video_embedding, p=2, dim=-1))
            dash_text = torch.exp(self.factor_text * t2p).unsqueeze(2)  # -> (a,k,1)
            dash_vis = torch.exp(self.factor_vis * v2p).unsqueeze(2)  # -> (b,k,1)

            results['text_word_features'] = (prob_text_mean.unsqueeze(1) + dash_text * prob_text_logsigma.unsqueeze(1))  # (b,M,dim)
            results['vis_patch_features'] = (prob_video_mean.unsqueeze(1) + dash_vis * prob_video_logsigma.unsqueeze(1))  # (b,M,dim)
            

            if is_train:
                logit_scale = self.clipmodel.logit_scale

                kl_loss = self.lambda1 * self.vib_loss(prob_video_embedding, prob_video_logsigma, prob_text_embedding, prob_text_logsigma)

                sim_matrix = torch.matmul(text_feat, vis_feat.t())
                t2v_label = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)
                t2v = torch.einsum('ad, bvd->abv', text_feat, prob_video_embedding)
                v2t = torch.einsum('bd, atd->abt', vis_feat, prob_text_embedding)
                t_alpha = self.evidence_compute(t2v)[0]
                v_alpha = self.evidence_compute(v2t)[0]

                ucn_loss1 = self.loss_uct(sim_matrix, t_alpha)
                ucn_loss2 = self.loss_uct(sim_matrix.T, v_alpha)
                ucn_loss = self.lambda2 * (ucn_loss1 + ucn_loss2) / 2
                

                results['prob_loss'] = kl_loss + ucn_loss

        else:
            video = video.reshape(-1, C, H, W)
            inputs = {"input_ids": text_input_ids,
                    "attention_mask": text_input_mask,
                    "pixel_values": video}
            outputs = self.clipmodel(**inputs)
            vis_features = outputs["vision_model_output"][1]

            vis_features = self.clipmodel.visual_projection(vis_features)
            vis_features = vis_features / vis_features.norm(dim=-1, keepdim=True)
            vis_features = vis_features.reshape(B, N, -1).mean(1)
            vis_features = vis_features / vis_features.norm(dim=-1, keepdim=True)
            
            results = {}
            results["text_features"] = outputs["text_embeds"]
            results["vis_features"] = vis_features
        if image is not None:
            B, img_num, C, H, W = image.shape
            L = caption_ids.shape[-1]
            inputs = {"input_ids": caption_ids.reshape(-1, L),
                    "attention_mask": caption_masks.reshape(-1, L),
                    "pixel_values": image.reshape(-1, 1, C, H, W),
                    "return_loss": False}
            outputs = self.clipmodel(**inputs)
            results["img_features"] = outputs["image_embeds"]
            results["cap_features"] = outputs["text_embeds"]
        
        return results
    
    def forward_video(self, video):
        inputs = {"pixel_values": video,
                "if_norm": True}
        video_features = self.clipmodel.get_image_features(**inputs)
        return video_features
    
    def forward_text(self, text_input_ids, text_input_mask):
        inputs = {"input_ids": text_input_ids,
                "attention_mask": text_input_mask,
                "if_norm": True}
        text_features = self.clipmodel.get_text_features(**inputs)
        return text_features

    def freeze_text_encoder(self, freeze_text_proj):
        freeze_list = [self.clipmodel.text_model]
        if freeze_text_proj:
            freeze_list.append(self.clipmodel.text_projection)
        for m in freeze_list:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
    def evidence_compute(self, sims):
        K = sims.size(1)
        E = torch.exp(sims / self.tau)
        # E = self.relu(sims)
        # E = self.softplus(sims)
        alpha = E + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        U = K / S

        return alpha, 1 - U

    def sim_proxy(self, text_feat, vis_feat, text_word_feat, vis_patch_feat, is_train=True):
        """

        :param text_embeds: (a,b,dim)
        :param vid_embeds: (b,dim)
        :return:
        """
        eps = 1e-7
        text_word_feat =  F.normalize(text_word_feat, p=2, dim=-1)
        vis_patch_feat =  F.normalize(vis_patch_feat, p=2, dim=-1)


        sims1 = torch.einsum('ad, bvd->abv', text_feat, vis_patch_feat).max(-1)[0]
        sims2 = torch.einsum('bd, atd->abt', vis_feat, text_word_feat).max(-1)[0]

        return (sims1 + sims2) / 2
    
    def probabilistic_video(self, video_pooled, videos):
        output = {}

        out, attn, residual = self.pie_net_vis(video_pooled, videos)        # (B 512) (B 12 512)   multiheadatt + fc + sigmoid + (residual) + laynorm
        output['attention'] = attn
        output['residual'] = residual       # B 512    

        uncertain_out = self.uncertain_net_vis(video_pooled, videos)        # (B 512) (B 12 512)   multiheadatt + fc + (residual)         
        logsigma = uncertain_out['logsigma']
        output['logsigma'] = logsigma       # B 512     可以看作是方差
        output['uncertainty_attention'] = uncertain_out['attention']

        out = l2_normalize(out)     # B 512     l2 normalization后 均值
        output['mean'] = out   

        output['embedding'] = sample_gaussian_tensors(out, logsigma, 7)      # B 7 512    从高斯分布中采样N个embedding  

        return output


    def probabilistic_text(self, text_pooled, text_token):
        output = {}

        out, attn, residual = self.pie_net_text(text_pooled, text_token)     # (B 512) (B 32 512)   multiheadatt + fc + sigmoid + (residual) + laynorm
        output['attention'] = attn
        output['residual'] = residual

        uncertain_out = self.uncertain_net_text(text_pooled, text_token)     # (B 512) (B 32 512)   multiheadatt + fc + (residual)   
        logsigma = uncertain_out['logsigma']
        output['logsigma'] = logsigma
        output['uncertainty_attention'] = uncertain_out['attention']

        out = l2_normalize(out)
        output['mean'] = out

        output['embedding'] = sample_gaussian_tensors(out, logsigma, 7)

        return output
    