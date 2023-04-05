import time

import numpy as np
import torch.nn as nn
import torch
from thop import profile, clever_format
from models.ENet.StripPooling import SPBlock


class InitialBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 relu=True):
        super().__init__()

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - As stated above the number of output channels for this
        # branch is the total minus 3, since the remaining channels come from
        # the extension branch
        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - 3,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias)

        # Extension branch
        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)

        # Initialize batch normalization to be used after concatenation
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)

        # Concatenate branches
        out = torch.cat((main, ext), 1)

        # Apply batch normalization
        out = self.batch_norm(out)

        return self.out_activation(out)


class RegularBottleneck(nn.Module):
    """Regular bottlenecks are the main building block of ENet.
    Main branch:
    1. Shortcut connection.

    Extension branch:
    1. 1x1 convolution which decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. regular, dilated or asymmetric convolution;
    3. 1x1 convolution which increases the number of channels back to
    ``channels``, also called an expansion;
    4. dropout as a regularizer.

    Keyword arguments:
    - channels (int): the number of input and output channels.
    - internal_ratio (int, optional): a scale factor applied to
    ``channels`` used to compute the number of
    channels after the projection. eg. given ``channels`` equal to 128 and
    internal_ratio equal to 2 the number of channels after the projection
    is 64. Default: 4.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer described above in item 2 of the extension
    branch. Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - dilation (int, optional): spacing between kernel elements for the
    convolution described in item 2 of the extension branch. Default: 1.
    asymmetric (bool, optional): flags if the convolution described in
    item 2 of the extension branch is asymmetric or not. Default: False.
    - dropout_prob (float, optional): probability of an element to be
    zeroed. Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.

    """

    def __init__(self,
                 channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True,
                 sp_layer=False):
        super().__init__()
        self.sp_layer = sp_layer
        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}."
                               .format(channels, internal_ratio))

        internal_channels = channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # 主分支
        # Main branch - shortcut connection
        # 扩展分支
        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution, and,
        # finally, a regularizer (spatial dropout). Number of channels is constant.

        # 1x1 projection convolution
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                channels,
                internal_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # If the convolution is asymmetric we split the main convolution in
        # two. Eg. for a 5x5 asymmetric convolution we have two convolution:
        # the first is 5x1 and the second is 1x5.
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation(),
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(channels), activation())
        if self.sp_layer:
            self.sp_block = SPBlock(inplanes=channels,outplanes=channels)

        self.ext_regul = nn.Dropout2d(p=dropout_prob)



        # PReLU layer to apply after adding the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut
        main = x

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        if self.sp_layer:
            ext = self.sp_block(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)


class DownsamplingBottleneck(nn.Module):
    """Downsampling bottlenecks further downsample the feature map size.

    Main branch:
    1. max pooling with stride 2; indices are saved to be used for
    unpooling later.

    Extension branch:
    1. 2x2 convolution with stride 2 that decreases the number of channels
    by ``internal_ratio``, also called a projection;
    2. regular convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer.

    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number of output channels.
    - internal_ratio (int, optional): a scale factor applied to ``channels``
    used to compute the number of channels after the projection. eg. given
    ``channels`` equal to 128 and internal_ratio equal to 2 the number of
    channels after the projection is 64. Default: 4.
    - return_indices (bool, optional):  if ``True``, will return the max
    indices along with the outputs. Useful when unpooling later.
    - dropout_prob (float, optional): probability of an element to be
    zeroed. Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 return_indices=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Store parameters that are needed later
        self.return_indices = return_indices

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.MaxPool2d(
            2,
            stride=2,
            return_indices=return_indices)

        # Extension branch - 2x2 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 2x2 projection convolution with stride 2
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # Convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(out_channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Main branch channel padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        # Before concatenating, check if main is on the CPU or GPU and
        # convert padding accordingly
        if main.is_cuda:
            padding = padding.cuda()

        # Concatenate
        main = torch.cat((main, padding), 1)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out), max_indices


class UpsamplingBottleneck(nn.Module):
    """The upsampling bottlenecks upsample the feature map resolution using max
    pooling indices stored from the corresponding downsampling bottleneck.

    Main branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. max unpool layer using the max pool indices from the corresponding
    downsampling max pool layer.

    Extension branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. transposed convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer.

    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number of output channels.
    - internal_ratio (int, optional): a scale factor applied to ``in_channels``
     used to compute the number of channels after the projection. eg. given
     ``in_channels`` equal to 128 and ``internal_ratio`` equal to 2 the number
     of channels after the projection is 64. Default: 4.
    - dropout_prob (float, optional): probability of an element to be zeroed.
    Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if ``True``.
    Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))

        # Remember that the stride is the same as the kernel_size, just like
        # the max pooling layers
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 1x1 projection convolution with stride 1
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation())

        # Transposed convolution
        self.ext_tconv1 = nn.ConvTranspose2d(
            internal_channels,
            internal_channels,
            kernel_size=2,
            stride=2,
            bias=bias)
        self.ext_tconv1_bnorm = nn.BatchNorm2d(internal_channels)
        self.ext_tconv1_activation = activation()

        # 1x1 expansion convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x, max_indices, output_size):
        # Main branch shortcut
        main = self.main_conv1(x)
        main = self.main_unpool1(
            main, max_indices, output_size=output_size)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(ext, output_size=output_size)
        ext = self.ext_tconv1_bnorm(ext)
        ext = self.ext_tconv1_activation(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)


class ENet(nn.Module):

    def __init__(self, num_classes, encoder_relu=False, decoder_relu=True,channels=[8,32,64],factor=2,sp_layer_bottle=True,sp_layer_mid=True):
        super().__init__()
        self.sp_layer_bottle = sp_layer_bottle
        self.sp_layer_mid = sp_layer_mid
        if self.sp_layer_mid:
            self.spm0 = SPBlock(inplanes=channels[0] * factor, outplanes=channels[0] * factor)
            self.spm1 = SPBlock(inplanes=channels[1] * factor, outplanes=channels[1] * factor)
            self.spm2 = SPBlock(inplanes=channels[2] * factor, outplanes=channels[2] * factor)

        self.initial_block = InitialBlock(3, channels[0]*factor, relu=encoder_relu)

        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(channels[0]*factor,
            channels[1]*factor,
            return_indices=True,
            dropout_prob=0.01,
            relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(
            channels[1]*factor, padding=1, dropout_prob=0.01, relu=encoder_relu,sp_layer=self.sp_layer_bottle)
        self.regular1_2 = RegularBottleneck(
            channels[1]*factor, padding=1, dropout_prob=0.01, relu=encoder_relu,sp_layer=self.sp_layer_bottle)
        self.regular1_3 = RegularBottleneck(
            channels[1]*factor, padding=1, dropout_prob=0.01, relu=encoder_relu,sp_layer=self.sp_layer_bottle)
        self.regular1_4 = RegularBottleneck(
            channels[1]*factor, padding=1, dropout_prob=0.01, relu=encoder_relu,sp_layer=self.sp_layer_bottle)


        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(
            channels[1]*factor,
            channels[2]*factor,
            return_indices=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(
            channels[2]*factor, padding=1, dropout_prob=0.1, relu=encoder_relu,sp_layer=self.sp_layer_bottle)
        self.dilated2_2 = RegularBottleneck(
            channels[2]*factor, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu,sp_layer=self.sp_layer_bottle)
        self.asymmetric2_3 = RegularBottleneck(
            channels[2]*factor,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu,
            sp_layer=self.sp_layer_bottle)
        self.dilated2_4 = RegularBottleneck(
            channels[2]*factor, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu,sp_layer=self.sp_layer_bottle)
        self.regular2_5 = RegularBottleneck(
            channels[2]*factor, padding=1, dropout_prob=0.1, relu=encoder_relu,sp_layer=self.sp_layer_bottle)
        self.dilated2_6 = RegularBottleneck(
            channels[2]*factor, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu,sp_layer=self.sp_layer_bottle)
        self.asymmetric2_7 = RegularBottleneck(
            channels[2]*factor,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu,sp_layer=self.sp_layer_bottle)
        self.dilated2_8 = RegularBottleneck(
            channels[2]*factor, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu,sp_layer=self.sp_layer_bottle)

        # Stage 3 - Encoder
        self.regular3_0 = RegularBottleneck(
            channels[2]*factor, padding=1, dropout_prob=0.1, relu=encoder_relu,sp_layer=self.sp_layer_bottle)
        self.dilated3_1 = RegularBottleneck(
            channels[2]*factor, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu,sp_layer=self.sp_layer_bottle)
        self.asymmetric3_2 = RegularBottleneck(
            channels[2]*factor,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu,sp_layer=self.sp_layer_bottle)
        self.dilated3_3 = RegularBottleneck(
            channels[2]*factor, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu,sp_layer=self.sp_layer_bottle)
        self.regular3_4 = RegularBottleneck(
            channels[2]*factor, padding=1, dropout_prob=0.1, relu=encoder_relu,sp_layer=self.sp_layer_bottle)
        self.dilated3_5 = RegularBottleneck(
            channels[2]*factor, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu,sp_layer=self.sp_layer_bottle)
        self.asymmetric3_6 = RegularBottleneck(
            channels[2]*factor,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu,sp_layer=self.sp_layer_bottle)
        self.dilated3_7 = RegularBottleneck(
            channels[2]*factor, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu,sp_layer=self.sp_layer_bottle)

        # Stage 4 - Decoder
        self.upsample4_0 = UpsamplingBottleneck(
            channels[2]*factor, channels[1]*factor, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(
            channels[1]*factor, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(
            channels[1]*factor, padding=1, dropout_prob=0.1, relu=decoder_relu)



        # Stage 5 - Decoder
        self.upsample5_0 = UpsamplingBottleneck(
            channels[1]*factor, channels[0]*factor, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(
            channels[0]*factor, padding=1, dropout_prob=0.1, relu=decoder_relu)


        self.transposed_conv = nn.ConvTranspose2d(
            channels[0]*factor,
            num_classes,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.init_weight()

    def forward(self, x):
        # Initial block
        input_size = x.size()
        # fm_input = x
        x = self.initial_block(x)
        # fm_init = x
        if self.sp_layer_mid:
            x = self.spm0(x)
        # Stage 1 - Encoder
        stage1_input_size = x.size()
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)
        # fm_stage1 = x
        if self.sp_layer_mid:
            x = self.spm1(x)
        # Stage 2 - Encoder
        stage2_input_size = x.size()
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)
        # fm_stage2 = x
        if self.sp_layer_mid:
            x = self.spm2(x)
        # Stage 3 - Encoder
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)
        # fm_stage3 = x
        if self.sp_layer_mid:
            x = self.spm2(x)
        # Stage 4 - Decoder
        x = self.upsample4_0(x, max_indices2_0, output_size=stage2_input_size)
        x = self.regular4_1(x)
        x = self.regular4_2(x)
        # fm_stage4 = x

        # Stage 5 - Decoder
        x = self.upsample5_0(x, max_indices1_0, output_size=stage1_input_size)
        x = self.regular5_1(x)
        # fm_stage5 = x

        # 反卷积输出原始尺寸
        out = self.transposed_conv(x, output_size=input_size)

        return out
        # tup  = (['fm_input',fm_input],
        #         ['fm_init',fm_init],
        #         ['fm_stage1',fm_stage1],
        #         ['fm_stage2',fm_stage2],
        #         ['fm_stage3',fm_stage3],
        #         ['fm_stage4',fm_stage4],
        #         ['fm_stage5',fm_stage5],
        #         ['out',out])
        # return dict(tup)
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


if __name__ == '__main__':
    # 定义网络
    model = ENet(num_classes=3,channels=[8,32,64],factor=1,sp_layer_bottle=True,sp_layer_mid=False).cuda()
    model.eval()
    # input大小
    input = torch.rand(1, 3, 640, 480).cuda()

    fps_list = []
    for i in range(500):
        time_start = time.time()
        out= model(input)
        time_end = time.time()
        fps_list.append(1 / (time_end - time_start))
    print('最快：{0:.4f}  >>>  最慢：{1:.4f}  >>>  平均：{2:.4f}'.format(np.max(fps_list), np.min(fps_list),
                                                                      np.mean(fps_list)))
    # # 分析模型复杂度
    flops, params, kmacs, omacs = profile(model, inputs=(input,))
    flops, params, kmacs1, omacs1 = clever_format([flops, params, kmacs, omacs], '%.3f')
    print('FLOPs: ', flops, 'params: ', params, 'KMacs', kmacs1, 'OMacs', omacs1)
    print('访存量：{}M'.format((kmacs + omacs)/1000000))