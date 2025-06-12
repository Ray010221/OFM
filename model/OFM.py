import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from segmentation_models_pytorch.decoders.unet import Unet


class ConfidenceNet(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super(ConfidenceNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv(x)
        return y


def rho_mask(sar_pred, opt_pred):
    ratio_sar = sar_pred - opt_pred
    ratio_opt = -ratio_sar
    return ratio_sar, ratio_opt

def oa_mask(label, num_classes, sar_classification, opt_classification, tau):
    label_one_hot = F.one_hot(label.squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2)

    score_sar = torch.sum(F.softmax(sar_classification / tau, dim=1) * label_one_hot, dim=1)
    score_opt = torch.sum(F.softmax(opt_classification / tau, dim=1) * label_one_hot, dim=1)

    return score_sar, score_opt

class ClusterDrop(nn.Module):
    def __init__(self, center_w=2, center_h=2):
        super().__init__()
        self.centers_proposal = nn.AdaptiveAvgPool2d((center_w, center_h))

    def forward(self, x, logit_mask):  # [b,c,w,h], [b,h,w]
        b, c, w, h = x.shape
        # Step 1: Initialize cluster centers
        centers = self.centers_proposal(x)  # [b,c,C_W,C_H]
        centers = rearrange(centers, 'b c w h -> b (w h) c')  # [b,M,c], where M = C_W * C_H

        # Step 2: Compute similarity between features and cluster centers
        x_flat = rearrange(x, 'b c w h -> b (w h) c')  # [b,N,c], where N = w * h
        sim = torch.sigmoid(pairwise_cos_sim(centers, x_flat)) # [b,M,N]

        # Step 3: Generate one-hot mask indicating which feature belongs to which cluster
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)  # [b,1,N]
        one_hot_mask = torch.zeros_like(sim)  # [b,M,N]
        one_hot_mask.scatter_(1, sim_max_idx, 1.0)  # Binary mask [b,M,N]

        # Step 4: Check logit_mask for each cluster center
        # Get the logit_mask value for each cluster center
        logit_mask_flat = rearrange(logit_mask, 'b w h -> b (w h)')  # [b,N]
        # logit_mask_flat = torch.relu(logit_mask_flat)
        cluster_logit_values = (one_hot_mask * logit_mask_flat.unsqueeze(1)).sum(dim=-1)  # [b,M]
        # Normalize by the number of points in each cluster
        cluster_logit_values = cluster_logit_values / (one_hot_mask.sum(dim=-1) + 1e-6)  # [b,M]

        # Step 5: Drop features based on cluster_logit_values using Bernoulli sampling
        # Convert cluster_logit_values to probabilities (assuming logit_mask is in [-1, 1])
        drop_prob = torch.relu(cluster_logit_values)  # [b,M]

        # Sample drop mask using Bernoulli distribution
        bernoulli_dist = torch.distributions.Bernoulli(1 - drop_prob)
        drop_mask = bernoulli_dist.sample()  # [b,M]
        drop_mask = drop_mask.unsqueeze(-1)  # [b,M,1]
        # Apply drop mask to features
        drop_mask = (one_hot_mask * drop_mask).sum(dim=1)  # [b,N]
        # drop_mask = drop_mask < 1  # [b,N]

        # Reshape back to original feature map
        out = rearrange(drop_mask, 'b (w h) -> b w h', w=w, h=h)
        return out

def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


class PDF(nn.Module):
    def __init__(
            self,
            sar_channels=1,
            opt_channels=3,
            num_classes=1000,
            backbone='resnet50',
            decoder_channels=(256, 128, 64, 32, 16),
    ):
        super(PDF, self).__init__()
        self.Unet_SAR = Unet(encoder_name=backbone,
                             encoder_depth=5,
                             encoder_weights="imagenet",
                             classes=num_classes,
                             in_channels=sar_channels,
                             decoder_channels=decoder_channels,
                             )

        self.Unet_OPT = Unet(encoder_name=backbone,
                             encoder_depth=5,
                             encoder_weights="imagenet",
                             classes=num_classes,
                             in_channels=opt_channels,
                             decoder_channels=decoder_channels,
                             )

        self.ConfidenceNet_SAR = ConfidenceNet(in_channels=decoder_channels[-1])
        self.ConfidenceNet_OPT = ConfidenceNet(in_channels=decoder_channels[-1])
        self.num_classes = num_classes
        self.cluster_drop = ClusterDrop(center_w=8, center_h=8)
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.mae_loss = nn.L1Loss(reduction='mean')

    def forward(self, sar, opt, label, stage):
        sar_features = self.Unet_SAR.encoder(sar)
        opt_features = self.Unet_OPT.encoder(opt)

        sar_decoder_output = self.Unet_SAR.decoder(*sar_features)
        opt_decoder_output = self.Unet_OPT.decoder(*opt_features)

        sar_decoder_output_cp = sar_decoder_output.clone().detach()
        opt_decoder_output_cp = opt_decoder_output.clone().detach()

        sar_masks = self.Unet_SAR.segmentation_head(sar_decoder_output)
        opt_masks = self.Unet_OPT.segmentation_head(opt_decoder_output)

        sar_tcp = self.ConfidenceNet_SAR(sar_decoder_output_cp)
        opt_tcp = self.ConfidenceNet_OPT(opt_decoder_output_cp)

        sar_holo = torch.log(opt_tcp) / (torch.log(sar_tcp * opt_tcp) + 1e-8)
        img_holo = torch.log(sar_tcp) / (torch.log(sar_tcp * opt_tcp) + 1e-8)

        w_all = torch.stack((sar_holo, img_holo), 1)
        softmax = nn.Softmax(1)
        w_all = softmax(w_all)
        w_sar = w_all[:, 0]
        w_opt = w_all[:, 1]

        fusion_masks = w_sar * sar_masks.detach() + w_opt * opt_masks.detach()

        # Train
        if stage == 'train':
            score_sar, score_opt = oa_mask(label, self.num_classes, sar_masks, opt_masks, 3)
            rho_sar, rho_opt = rho_mask(sar_tcp, opt_tcp)
            sar_drop_mask = self.cluster_drop(sar_decoder_output_cp, rho_sar.squeeze(1))
            opt_drop_mask = self.cluster_drop(opt_decoder_output_cp, rho_opt.squeeze(1))

            loss1 = self.ce_loss(fusion_masks, label) + self.ce_loss(torch.mul(sar_masks, sar_drop_mask.unsqueeze(1)), label) + self.ce_loss(torch.mul(opt_masks, opt_drop_mask.unsqueeze(1)), label)
            loss2 = self.mae_loss(sar_tcp.squeeze(1), score_sar.detach()) + self.mae_loss(opt_tcp.squeeze(1), score_opt.detach())

            return fusion_masks, loss1, loss2
        # Test
        return fusion_masks, sar_masks, opt_masks


if __name__ == '__main__':
    model = PDF(num_classes=6).cuda()

    sar_img = torch.randn(4, 1, 256, 256).cuda()
    opt_img = torch.randn(4, 3, 256, 256).cuda()
    label = torch.randint(0, 6, (4, 256, 256)).cuda()
    stage = 'trian'

    fusion_logits, sar_logits, opt_logits = model(sar_img, opt_img, label, stage)
