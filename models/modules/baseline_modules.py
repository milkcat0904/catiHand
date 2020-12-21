import torch.nn as nn

class Backbone(nn.Module):
    def __init__(self, hyper):
        super(Backbone, self).__init__()

        self.use_mvn = hyper['model']['mvn_layer']
        self.fc_input = hyper['model']['flatten']
        self.num_cell = hyper['model']['num_cell'] # 128
        self.num_joint = hyper['model']['num_joint']
        self.input_channel = hyper['model']['input_channel'] # [1, 24, 48, 98]
        self.conv_kernel = hyper['model']['kernel_size']
        self.point2d_cls_layer = hyper['model']['point2d_classifier']

        self.addon_dim = hyper['model']['addon_dim']
        self.output_dim = hyper['model']['output_dim']
        self.use_bn = hyper['model']['bn']
        self.use_cls = hyper['model']['use_cls']
        self.rot_ratio = hyper['train']['rot_loss_ratio']
        self.use_heatmap = hyper['model']['heatmap']['use_heatmap']
        self.heatmap_size = hyper['model']['heatmap']['heatmap_shape']

        # 网络
        if self.use_mvn:
            self.mvn = MVN()
        self.stage1 = self.make_block(self.input_channel[0], self.input_channel[1], pool_mode = 'ave')
        self.stage2 = self.make_block(self.input_channel[1], self.input_channel[2], pool_mode = 'max')
        self.stage3 = self.make_block(self.input_channel[2], self.input_channel[3], pool_mode = 'max', out_mode = 'list')

        for i in range(hyper['model']['num_block']-1):
            tmpLayer = self.make_block(self.input_channel[3], self.input_channel[3], pool_mode = 'max', out_mode = 'list')
            self.stage3.extend(tmpLayer)
        self.stage3 = nn.Sequential(*self.stage3)

    def make_block(self, input_channel, output_channel, pool_mode='max', out_mode = 'seq'):
        if self.use_bn:
            stage_network = [nn.Conv2d(input_channel, output_channel,
                                       kernel_size = self.conv_kernel, padding = 1),
                             nn.BatchNorm2d(output_channel),
                             nn.ReLU(True),
                             nn.Conv2d(output_channel, output_channel,
                                       kernel_size = self.conv_kernel, padding = 1),
                             nn.BatchNorm2d(output_channel),
                             nn.ReLU(True)]
        else:
            stage_network = [nn.Conv2d(input_channel, output_channel,
                                       kernel_size = self.conv_kernel, padding = 1),
                             nn.ReLU(True),
                             nn.Conv2d(output_channel, output_channel,
                                       kernel_size = self.conv_kernel, padding = 1),
                             nn.ReLU(True)]
        if pool_mode == 'ave':
            stage_network.append(nn.AvgPool2d(kernel_size = 2, stride = 2))
        elif pool_mode == 'max':
            stage_network.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
        else:
            pass

        if out_mode == 'list':
            return stage_network
        else:
            return nn.Sequential(*stage_network)

    # 2d 没有addon项目 backbone fm输出大小2x2 channel数是 input_channel[-1]
    # (2 + 2*0 -1 )/1 + 1
    def conv_regression(self, input_channel, output_channel, output_dim):

        out = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size = 1),
            # nn.BatchNorm2d(output_channel),
            nn.ReLU(True),
            nn.Conv2d(output_channel, output_channel, kernel_size = 1),
            # nn.BatchNorm2d(output_channel),
            nn.ReLU(True),
            # nn.Sigmoid(),
            nn.Conv2d(output_channel, output_dim, kernel_size = 1), #
            # nn.BatchNorm2d(output_dim),
            nn.AvgPool2d(kernel_size = 2, stride = 2)
        )
        return out

    def heatmap_regression(self, input_channel,  output_dim=21):
        out = nn.Sequential(
            nn.ConvTranspose2d(input_channel, 48, kernel_size = 3),
            nn.ReLU(True),
            nn.ConvTranspose2d(48, 24, kernel_size = 2, stride = 2),
            nn.ReLU(True),
            # nn.Sigmoid(),
            nn.ConvTranspose2d(24, 24, kernel_size = 3),  #
            nn.ReLU(True),
            nn.ConvTranspose2d(24, output_dim, kernel_size = 2, stride = 2)
        )
        return out

    def fc_regression(self,  output_dim, addon_l = 0):
        input_num = self.fc_input + addon_l
        if self.use_bn:
            cls = nn.Sequential(nn.Linear(input_num, self.num_cell),
                                nn.BatchNorm1d(self.num_cell),
                                nn.ReLU(True),
                                nn.Linear(self.num_cell, self.num_cell),
                                nn.BatchNorm1d(self.num_cell),
                                nn.ReLU(True),
                                nn.Linear(self.num_cell, output_dim))
        else:
            cls = nn.Sequential(nn.Linear(input_num, self.num_cell),
                                nn.ReLU(True),
                                nn.Linear(self.num_cell, self.num_cell),
                                nn.ReLU(True),
                                nn.Linear(self.num_cell, output_dim))
        return cls

    def classifier(self,  output_dim, input_num):
        if self.use_bn:
            cls = nn.Sequential(nn.Linear(input_num, self.num_cell),
                                nn.BatchNorm1d(self.num_cell),
                                nn.ReLU(True),
                                nn.BatchNorm1d(self.num_cell),
                                nn.ReLU(True),
                                nn.Linear(self.num_cell, output_dim))
        else:
            cls = nn.Sequential(nn.Linear(input_num, self.num_cell),
                                nn.ReLU(True),
                                nn.Linear(self.num_cell, self.num_cell),
                                nn.ReLU(True),
                                nn.Linear(self.num_cell, output_dim))
        return cls

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            #
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class MVN(nn.Module):
    """Mean-Variance Normalization (MVN) Layer
    """
    def __init__(self, normalize_variance=True, across_channels=False, eps=None):
        super(MVN, self).__init__()
        self.normalize_variance = normalize_variance
        self.across_channels = across_channels
        if eps is None:
            eps = 1e-9
        self.eps = eps

    def extra_repr(self):
        """Extra information
        """
        return "eps={}{}{}".format(
            self.eps,
            ", normalize_variance" if self.normalizze_variance else "",
            ", across_channels" if self.across_channels else "",
        )

    def forward(self, x):
        std = None
        shape = x.data.shape
        n_b = shape[0]

        # use channels
        if self.across_channels:
            if shape[0] == 1:
                # single batch normalization reduction
                mean = x.mean()
                if self.normalize_variance:
                    std = x.std()
            else:
                mean = x.view(n_b, -1).mean(1).view(n_b, 1, 1, 1)
                if self.normalize_variance:
                    std = x.view(n_b, -1).std(1).view(n_b, 1, 1, 1)

        else:
            n_c = shape[1]
            mean = std = x.view(n_b, n_c, -1).mean(2).view(n_b, n_c, 1, 1)
            if self.normalize_variance:
                std = x.view(n_b, n_c, -1).std(2).view(n_b, n_c, 1, 1)
        if std is not None:
            return (x - mean) / (std + self.eps)
        return x - mean
