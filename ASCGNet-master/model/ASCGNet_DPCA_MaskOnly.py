
import math
import torch
from torch import nn
import torch.nn.functional as F
import model.resnet as models
import model.vgg as vgg_models


def rgb_to_hsv_differentiable(rgb_image):
    rgb_clamped = torch.clamp(rgb_image, 0.0, 1.0)
    r, g, b = rgb_clamped[:, 0:1, :, :], rgb_clamped[:, 1:2, :, :], rgb_clamped[:, 2:3, :, :]
    max_val, _ = torch.max(rgb_clamped, dim=1, keepdim=True)
    min_val, _ = torch.min(rgb_clamped, dim=1, keepdim=True)
    delta = max_val - min_val + 1e-7
    hue = torch.zeros_like(max_val)
    mask_r = (max_val == r)
    mask_g = (max_val == g)
    mask_b = (max_val == b)
    hue[mask_r] = ((g - b) / delta % 6)[mask_r]
    hue[mask_g] = ((b - r) / delta + 2)[mask_g]
    hue[mask_b] = ((r - g) / delta + 4)[mask_b]
    hue = hue / 6.0
    saturation = torch.where(max_val == 0, torch.zeros_like(delta), delta / (max_val + 1e-7))
    value = max_val
    return torch.cat([hue, saturation, value], dim=1)


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


def get_vgg16_layer(model):
    layer0_idx, layer1_idx = range(0, 7), range(7, 14)
    layer2_idx, layer3_idx = range(14, 24), range(24, 34)
    layer4_idx = range(34, 43)
    layers_0, layers_1, layers_2, layers_3, layers_4 = [], [], [], [], []
    for idx in layer0_idx: layers_0 += [model.features[idx]]
    for idx in layer1_idx: layers_1 += [model.features[idx]]
    for idx in layer2_idx: layers_2 += [model.features[idx]]
    for idx in layer3_idx: layers_3 += [model.features[idx]]
    for idx in layer4_idx: layers_4 += [model.features[idx]]
    return nn.Sequential(*layers_0), nn.Sequential(*layers_1), nn.Sequential(*layers_2), nn.Sequential(
        *layers_3), nn.Sequential(*layers_4)


class DynamicPolarColorAnchor(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(1))
        self.proto_alpha = nn.Parameter(torch.zeros(1))

    def hsv_to_polar_coord(self, hsv_tensor):
        h, s, v = hsv_tensor[:, 0:1, :, :], hsv_tensor[:, 1:2, :, :], hsv_tensor[:, 2:3, :, :]
        angle = h * 2.0 * math.pi
        x = s * torch.cos(angle)
        y = s * torch.sin(angle)
        z = v
        return torch.cat([x, y, z], dim=1)

    def forward(self, q_img, s_x, mask_list, corr_query_mask, feat_size):
        q_img_small = F.interpolate(q_img, size=feat_size, mode='area')
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(q_img.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(q_img.device)
        q_rgb = torch.clamp(q_img_small * std + mean, 0.0, 1.0)
        q_hsv = rgb_to_hsv_differentiable(q_rgb)
        q_polar = self.hsv_to_polar_coord(q_hsv)

        s_protos = []
        for i in range(len(mask_list)):
            s_img_small = F.interpolate(s_x[:, i, :, :, :], size=feat_size, mode='area')
            s_mask_small = F.interpolate(mask_list[i], size=feat_size, mode='nearest')
            s_rgb = torch.clamp(s_img_small * std + mean, 0.0, 1.0)
            s_hsv = rgb_to_hsv_differentiable(s_rgb)
            s_polar = self.hsv_to_polar_coord(s_hsv)

            valid_s = s_mask_small.sum(dim=(2, 3), keepdim=True) + 1e-7
            s_proto = (s_polar * s_mask_small).sum(dim=(2, 3), keepdim=True) / valid_s
            s_protos.append(s_proto)
        support_color_proto = torch.stack(s_protos, dim=1).mean(dim=1)


        q_pseudo_mask = corr_query_mask.detach()
        valid_q = q_pseudo_mask.sum(dim=(2, 3), keepdim=True) + 1e-7
        query_color_proto = (q_polar * q_pseudo_mask).sum(dim=(2, 3), keepdim=True) / valid_q

        alpha = torch.sigmoid(self.proto_alpha)
        dynamic_proto = alpha * support_color_proto + (1.0 - alpha) * query_color_proto

        dist_sq = torch.sum((q_polar - dynamic_proto) ** 2, dim=1, keepdim=True)
        sigma_sq = torch.exp(self.log_sigma) ** 2 + 1e-5
        color_prior_map = torch.exp(-dist_sq / (2 * sigma_sq))

        return color_prior_map


class ASCGNet_DPCA_MaskOnly(nn.Module):
    def __init__(self, layers=50, classes=2, zoom_factor=8, \
                 criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
                 pretrained=True, shot=1, ppm_scales=[60, 30, 15, 8], vgg=False):
        super(ASCGNet_DPCA_MaskOnly, self).__init__()
        assert layers in [50, 101, 152]
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.shot = shot
        self.vgg = vgg
        models.BatchNorm = BatchNorm
        self.ppm_scales = ppm_scales

        if self.vgg:
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=pretrained)
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = get_vgg16_layer(vgg16)
        else:
            if layers == 50:
                resnet = models.resnet50(pretrained=pretrained)
            elif layers == 101:
                resnet = models.resnet101(pretrained=pretrained)
            else:
                resnet = models.resnet152(pretrained=pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                        resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        reduce_dim = 256
        fea_dim = 512 + 256 if self.vgg else 1024 + 512

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1), nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False), nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False), nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))

        self.avgpool_list = nn.ModuleList([nn.AdaptiveAvgPool2d(bin) for bin in self.ppm_scales if bin > 1])

        mask_add_num = 1
        self.init_merge, self.beta_conv, self.inner_cls = [], [], []

        for idx, bin in enumerate(self.ppm_scales):
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim * 2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))

            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1), nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))

        self.init_merge = nn.ModuleList(self.init_merge)
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)

        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim * len(self.ppm_scales), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True))

        self.alpha_conv = nn.ModuleList([
            nn.Sequential(nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, stride=1, padding=0, bias=False),
                          nn.ReLU())
            for idx in range(len(self.ppm_scales) - 1)
        ])

        self.polar_color_anchor = DynamicPolarColorAnchor()

    def forward(self, x, s_x=torch.FloatTensor(1, 1, 3, 473, 473).cuda(), s_y=torch.FloatTensor(1, 1, 473, 473).cuda(),
                y=None):
        x_size = x.size()
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2), query_feat_3.size(3)),
                                             mode='bilinear', align_corners=True)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)

        supp_feat_list, final_supp_list, mask_list = [], [], []
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:, i, :, :, :])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear',
                                     align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3 * mask)
                final_supp_list.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2), supp_feat_3.size(3)),
                                                mode='bilinear', align_corners=True)
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_supp(supp_feat)
            supp_feat = Weighted_GAP(supp_feat, mask)
            supp_feat_list.append(supp_feat)

        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)
            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
            q = query_feat_4
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]
            tmp_query = q.contiguous().view(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)
            tmp_supp = s.contiguous().view(bsize, ch_sz, -1).permute(0, 2, 1)
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)
            similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
            similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                    similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query_mask_list.append(F.interpolate(similarity.view(bsize, 1, sp_sz, sp_sz),
                                                      size=(query_feat_3.size()[2], query_feat_3.size()[3]),
                                                      mode='bilinear', align_corners=True))

        corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)
        corr_query_mask = F.interpolate(corr_query_mask, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear',
                                        align_corners=True)

        if self.shot > 1:
            supp_feat = sum(supp_feat_list) / len(supp_feat_list)
        else:
            supp_feat = supp_feat_list[0]

        color_anchor_map = self.polar_color_anchor(
            x, s_x, mask_list, corr_query_mask, feat_size=(query_feat.size(2), query_feat.size(3))
        )

        refined_corr_mask = corr_query_mask * color_anchor_map

        out_list = []
        pyramid_feat_list = []

        for idx, tmp_bin in enumerate(self.ppm_scales):
            if tmp_bin <= 1.0:
                bin = int(query_feat.shape[2] * tmp_bin)
                query_feat_bin = nn.AdaptiveAvgPool2d(bin)(query_feat)
            else:
                bin = tmp_bin
                query_feat_bin = self.avgpool_list[idx](query_feat)

            supp_feat_bin = supp_feat.expand(-1, -1, bin, bin)
            corr_mask_bin = F.interpolate(refined_corr_mask, size=(bin, bin), mode='bilinear', align_corners=True)

            merge_feat_bin = torch.cat([query_feat_bin, supp_feat_bin, corr_mask_bin], 1)
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)


            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx - 1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx - 1](rec_feat_bin) + merge_feat_bin

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)),
                                           mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)
            out_list.append(inner_out_bin)

        query_feat = torch.cat(pyramid_feat_list, 1)
        query_feat = self.res1(query_feat)
        query_feat = self.res2(query_feat) + query_feat
        out = self.cls(query_feat)

        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            main_loss = self.criterion(out, y.long())
            aux_loss = torch.zeros_like(main_loss).cuda()
            for idx_k in range(len(out_list)):
                inner_out = out_list[idx_k]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss = aux_loss + self.criterion(inner_out, y.long())
            aux_loss = aux_loss / len(out_list)
            return out.max(1)[1], main_loss, aux_loss
        else:
            return out