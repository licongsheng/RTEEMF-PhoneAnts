# implement the Wavenet architecture as the backbone for the EEG Diffusion model
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


def passthrough(x, **kwargs):
    return x


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=[1, 5, 5], padding=[0, 2, 2])
        self.bn1 = nn.BatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.conv1(x)
        out = self.bn1(out)
        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 1)
        out = self.relu1(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class BackBone(nn.Module):
    def __init__(self, elu=True, nll=False, feature_channels=8192):
        super(BackBone, self).__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        return out256


class OTAEstimator(nn.Module):
    def __init__(self, in_channels=8192, tis_channels=1, eirp_channels=33*64):
        super(OTAEstimator, self).__init__()
        self.fc_1 = nn.Linear(in_channels+1, 4096)
        self.fc_eirp = nn.Linear(4096, eirp_channels)
        self.fc_2 = nn.Linear(4096, 256)
        self.fc_tis = nn.Linear(256, tis_channels)

    def forward(self, x, fre):
        f1 = self.fc_1(torch.cat([x, fre], dim=1))
        eirp = self.fc_eirp(f1)
        #eirp = F.relu(eirp)

        f2 = self.fc_2(f1)
        tis = self.fc_tis(f2)
        return eirp, tis


class SAREstimator(nn.Module):
    def __init__(self, in_channels=8192, res_channels=12):
        super(SAREstimator, self).__init__()
        self.fc_1 = nn.Linear(in_channels+1, 1024)
        self.fc_2 = nn.Linear(1024, 256)
        self.fc_3 = nn.Linear(256, res_channels)

    def forward(self, x, fre):
        f1 = self.fc_1(torch.cat([x, fre], dim=1))
        f2 = self.fc_2(f1)
        sars = self.fc_3(f2)
        #sars = F.relu(sars)
        return sars


def get_inplanes():
    return [16, 16, 32, 64]
    #return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=3,
                 conv1_t_stride=2,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=8192):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 5, 5),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 1, 1),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.fc = nn.Linear(8064, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        #
        #x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        #x = self.fc(x)
        return x


class RTEEMFAntPerformace(nn.Module):
    def __init__(self, model_depth=50, n_input_channels=1, n_features=13440, n_classes=8192): # 13440 n_input_channels = 1 (magntidue form) or 6 (complex form)
        super(RTEEMFAntPerformace, self).__init__()
        assert model_depth in [10, 18, 34, 50, 101, 152, 200]
        if model_depth == 10:
            self.backbone = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), n_input_channels=n_input_channels, n_classes=n_classes)
        elif model_depth == 18:
            self.backbone = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), n_input_channels=n_input_channels, n_classes=n_classes)
        elif model_depth == 34:
            self.backbone = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), n_input_channels=n_input_channels, n_classes=n_classes)
        elif model_depth == 50:
            self.backbone = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), n_input_channels=n_input_channels, n_classes=n_classes)
        elif model_depth == 101:
            self.backbone = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), n_input_channels=n_input_channels, n_classes=n_classes)
        elif model_depth == 152:
            self.backbone = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), n_input_channels=n_input_channels, n_classes=n_classes)
        elif model_depth == 200:
            self.backbone = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), n_input_channels=n_input_channels, n_classes=n_classes)
        self.sar_estimation = SAREstimator(in_channels=n_features)
        self.ota_fs_estimation = OTAEstimator(in_channels=n_features)
        self.ota_Lhand_estimation = OTAEstimator(in_channels=n_features)
        self.ota_Rhand_estimation = OTAEstimator(in_channels=n_features)
        self.ota_Lhead_estimation = OTAEstimator(in_channels=n_features)
        self.ota_Rhead_estimation = OTAEstimator(in_channels=n_features)
        self.ota_Lhh_estimation = OTAEstimator(in_channels=n_features)
        self.ota_Rhh_estimation = OTAEstimator(in_channels=n_features)

    def forward(self, x, fre):
        feature_1D = self.backbone(x)
        sar = self.sar_estimation(feature_1D, fre)
        eirp_fs, tis_fs = self.ota_fs_estimation(feature_1D, fre)
        eirp_Lhand, tis_Lhand = self.ota_Lhand_estimation(feature_1D, fre)
        eirp_Rhand, tis_Rhand = self.ota_Rhand_estimation(feature_1D, fre)
        eirp_Lhead, tis_Lhead = self.ota_Lhead_estimation(feature_1D, fre)
        eirp_Rhead, tis_Rhead = self.ota_Rhead_estimation(feature_1D, fre)
        eirp_Lhh, tis_Lhh = self.ota_Lhh_estimation(feature_1D, fre)
        eirp_Rhh, tis_Rhh = self.ota_Rhh_estimation(feature_1D, fre)
        return sar, torch.cat([tis_fs, tis_Lhand, tis_Rhand, tis_Lhead, tis_Rhead, tis_Lhh, tis_Rhh], dim=1),\
               torch.cat([eirp_fs.unsqueeze(1), eirp_Lhand.unsqueeze(1), eirp_Rhand.unsqueeze(1), eirp_Lhead.unsqueeze(1), eirp_Rhead.unsqueeze(1), eirp_Lhh.unsqueeze(1), eirp_Rhh.unsqueeze(1)], dim=1)

    def require_grad_model_params(self, backbone=True, sar=True, fs_ota=True, lhead_ota=True, rhead_ota=True,
                           lhand_ots=True, rhand_ots=True, lhh_ota=True, rhh_ota=True):
        self.fixed_backbone(backbone)
        self.fixed_sar_estimation(sar)
        self.fixed_ota_fs_estimation(fs_ota)
        self.fixed_ota_Lhand_estimation(lhand_ots)
        self.fixed_ota_Rhand_estimation(rhand_ots)
        self.fixed_ota_Lhead_estimation(lhead_ota)
        self.fixed_ota_Rhead_estimation(rhand_ots)
        self.fixed_ota_Lhh_estimation(lhh_ota)
        self.fixed_ota_Rhh_estimation(rhh_ota)

    def fixed_backbone(self, required=False):
        for param in self.backbone.parameters():
            param.requires_grad = required

    def fixed_sar_estimation(self, required=False):
        for param in self.sar_estimation.parameters():
            param.requires_grad = required

    def fixed_ota_fs_estimation(self, required=False):
        for param in self.ota_fs_estimation.parameters():
            param.requires_grad = required

    def fixed_ota_Lhand_estimation(self, required=False):
        for param in self.ota_Lhand_estimation.parameters():
            param.requires_grad = required

    def fixed_ota_Rhand_estimation(self, required=False):
        for param in self.ota_Rhand_estimation.parameters():
            param.requires_grad = required

    def fixed_ota_Lhead_estimation(self, required=False):
        for param in self.ota_Lhead_estimation.parameters():
            param.requires_grad = required

    def fixed_ota_Rhead_estimation(self, required=False):
        for param in self.ota_Rhead_estimation.parameters():
            param.requires_grad = required

    def fixed_ota_Lhh_estimation(self, required=False):
        for param in self.ota_Lhh_estimation.parameters():
            param.requires_grad = required

    def fixed_ota_Rhh_estimation(self, required=False):
        for param in self.ota_Rhh_estimation.parameters():
            param.requires_grad = required


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fre = (918-600)/(6000-600)
    model = RTEEMFAntPerformace(model_depth=34, n_input_channels=1).to(device)
    pseudo_input = torch.randn(1, 1, 79, 143, 295).to(device)
    sar, tiss, eirps = model(pseudo_input, torch.tensor([fre]).to(device))
    print(sar.shape, tiss.shape, eirps.shape)
    torch.save(model, "./checkpoints/RTEEMFAntPerf.pth")
