import torch
import torch.nn as nn
from transformers import CLIPModel
import numpy as np
from OT.my_ot_layer import OTLayer
import torch.nn.functional as F


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X



class HateModel(nn.Module):
    def __init__(self, scratch=True, class_num = 3, gamma=0.7,pretrained_weights_path = '../clip-vit-base-patch32'):
        super(HateModel, self).__init__()
        #去除了position——embedding
        self.pre_model = CLIPModel.from_pretrained(pretrained_weights_path,ignore_mismatched_sizes=True)
        self.scratch = scratch
        self.img_trans = nn.Linear(768,512)
        self.fc = nn.Linear(1024,256)
        self.similarity_module = EncoderSimilarity(embed_size=512, sim_dim=256,class_num = class_num)
        self.gamma = gamma
        self.fc1 = nn.Linear(256, class_num)
        # if class_num ==2:
        #     self.fc1 = nn.Linear(256, 2)
        # elif class_num == 3:
        #     self.fc1 = nn.Linear(256, 3)
        # else:
        #     self.fc1 = nn.Linear(256, 4)
        if scratch:
            for params in self.pre_model.parameters():
                params.requires_grad = True

        else:
            for params in self.pre_model.parameters():
                params.requires_grad = False



    def forward(self, img, input_ids, att_mask):
        # patches, img_embed= self.pre_model.get_image_features(img,output_hidden_states = True, return_dict=True)


        output_attentions = self.pre_model.config.output_attentions

        vision_outputs = self.pre_model.vision_model(
            pixel_values=img,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=False,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_embeds = self.pre_model.visual_projection(pooled_output)
        patches = vision_outputs[0]


        text_outputs = self.pre_model.text_model(
            input_ids=input_ids,
            attention_mask=att_mask,
            output_hidden_states=True,
        )


        text_embeds = text_outputs[1]
        text_embeds = self.pre_model.text_projection(text_embeds)
        text_seq = text_outputs.last_hidden_state

        # normalized features
        img_embed = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embed = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)



        patches = self.img_trans(patches)
        # sequence,text_embed = self.pre_model.get_text_features(input_ids.squeeze(1), attention_mask=att_mask.squeeze(1),output_hidden_states = True)
        # sequence, text_embed = self.pre_model.get_text_features(input_ids,
        #                                                             attention_mask=att_mask,output_hidden_states=True)
        sim_vec = self.similarity_module(patches,text_seq,img_embed,text_embed,cap_lens = 77)

        multimodal_rp = torch.cat((text_embed,img_embed), dim =1)
        logits = self.fc(multimodal_rp)
        logits = self.fc1(logits)
        gamma = self.gamma

        score = gamma * logits + (1 - gamma) * sim_vec
        # score = logits
        # score = gamma * logits + (1 - gamma) * sim_vec
        # for cross_attention
        # score = gamma * logits + (1 - gamma) * sim_vec.squeeze(dim=1)
        # return logits
        return score


class GraphReasoning(nn.Module):
    """
    Perform the similarity graph reasoning with a full-connected graph
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_sgr: reasoned graph nodes after several steps, shape: (batch_size, L+1, 256)
    """
    def __init__(self, sim_dim):
        super(GraphReasoning, self).__init__()

        self.graph_query_w = nn.Linear(sim_dim, sim_dim)
        self.graph_key_w = nn.Linear(sim_dim, sim_dim)
        self.sim_graph_w = nn.Linear(sim_dim, sim_dim)
        self.relu = nn.ReLU()

        self.init_weights()

    def forward(self, sim_emb):
        sim_query = self.graph_query_w(sim_emb)
        sim_key = self.graph_key_w(sim_emb)
        sim_edge = torch.softmax(torch.bmm(sim_query, sim_key.permute(0, 2, 1)), dim=-1)
        sim_sgr = torch.bmm(sim_edge, sim_emb)
        sim_sgr = self.relu(self.sim_graph_w(sim_sgr))
        return sim_sgr

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def SCAN_attention(query, context, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch, queryL, sourceL
    attn = F.softmax(attn*smooth, dim=2)

    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)
    weightedContext = l2norm(weightedContext, dim=-1)

    return weightedContext
class EncoderSimilarity(nn.Module):
    """
    Compute the image-text similarity by SGR, SAF, AVE
    Args: - img_emb: local region embeddings, shape: (batch_size, 36, 1024)
          - cap_emb: local word embeddings, shape: (batch_size, L, 1024)
    Returns:
        - sim_all: final image-text similarities, shape: (batch_size, batch_size).
    """
    def __init__(self, embed_size, sim_dim,class_num, module_name='SGR', sgr_step=4):
        super(EncoderSimilarity, self).__init__()
        self.module_name = module_name

        # self.v_global_w = VisualSA(embed_size, 0.4, 36)
        # self.t_global_w = TextSA(embed_size, 0.4)

        self.sim_tranloc_w = nn.Linear(embed_size, sim_dim)
        self.sim_tranglo_w = nn.Linear(embed_size, sim_dim)

        self.sim_eval_w = nn.Linear(sim_dim, class_num)
        self.sigmoid = nn.Sigmoid()

        #best_settings:max_iter = 5
        self.text2img_ot = OTLayer(in_dim=512, out_size=50, heads=1, eps=0.1, max_iter=3,\
                                   position_encoding=None, position_sigma=0.1, out_dim=None,\
                                   dropout=0.4)
        self.img2text_ot = OTLayer(in_dim=512, out_size=77, heads=1, eps=0.1, max_iter=3,\
                                   position_encoding=None, position_sigma=0.1, out_dim=None,\
                                   dropout=0.4)


        if module_name == 'SGR':
            self.SGR_module = nn.ModuleList([GraphReasoning(sim_dim) for i in range(sgr_step)])
        # elif module_name == 'SAF':
        #     self.SAF_module = AttentionFiltration(sim_dim)
        else:
            raise ValueError('Invalid input of opt.module_name in opts.py')

        self.init_weights()

    def forward(self, img_emb, cap_emb, img_g,text_g, cap_lens):
        sim_all = []
        n_image = img_emb.size(0)
        n_caption = cap_emb.size(0)

        # get enhanced global images by self-attention
        img_ave = torch.mean(img_emb, 1)
        # img_glo = self.v_global_w(img_emb, img_ave)

        for i in range(n_caption):
            # get the i-th sentence
            # n_word = cap_lens[i]
            n_word = cap_lens
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            # get enhanced global i-th text by self-attention
            cap_ave_i = torch.mean(cap_i, 1)
            # cap_glo_i = self.t_global_w(cap_i, cap_ave_i)
            img_i = img_emb[i].unsqueeze(dim = 0)
            # local-global alignment construction
            # Context_img = SCAN_attention(cap_i, img_emb[i].unsqueeze(dim = 0), smooth=9.0)
            # Context_text = SCAN_attention(img_emb[i].unsqueeze(dim = 0), cap_i, smooth=9.0)
            # Context_img = torch.ones_like(cap_i_expand)
            Context_img = self.text2img_ot(cap_i, img_i)
            Context_text = self.img2text_ot(img_i,cap_i)


            # for heuristic
            # sim_loc1 = torch.pow(torch.sub(Context_img, cap_i), 2)
            # sim_loc2 = torch.pow(torch.sub(Context_text, img_i), 2)
            sim_loc = torch.cat((Context_img, Context_text), dim=1)

            # sim_loc = torch.cat((Context_img, Context_text), dim=1)
            sim_loc = l2norm(self.sim_tranloc_w(sim_loc), dim=-1)

            sim_emb = sim_loc

            # sim_glo = torch.pow(torch.sub(img_glo, cap_glo_i), 2)
            # sim_glo = l2norm(self.sim_tranglo_w(sim_glo), dim=-1)

            # concat the global and local alignments
            # sim_emb = torch.cat([sim_glo.unsqueeze(1), sim_loc], 1)

            # compute the final similarity vector
            if self.module_name == 'SGR':
                for module in self.SGR_module:
                    sim_emb = module(sim_emb)
                sim_vec = sim_emb[:, 0, :]
            else:
                sim_vec = self.SAF_module(sim_emb)

            # compute the final similarity score
            # sim_i = self.sigmoid(self.sim_eval_w(sim_vec))[i]
            # for cross attention
            sim_i = self.sigmoid(self.sim_eval_w(sim_vec))
            sim_all.append(sim_i)

        # (n_image, n_caption)
        # sim_all = torch.stack(sim_all, 0)
        sim_all = torch.stack(sim_all, 0).squeeze(dim = 1)

        return sim_all

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



a = 1
