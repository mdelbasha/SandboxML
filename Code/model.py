from keras.models import Model
from keras.models import Input
from keras.layers import Conv3D
from keras.layers import Add
from keras.layers import concatenate
from keras.layers import Conv3DTranspose
from keras.activations import softmax


# This is first try of making the V-net from the paper without
# looking online to see exactly what is done to make it work
def get_VNet(input_size=128, samples=32):
    num_filters = [16, 32, 64, 128, 256]  # number of filters used
    lg_kernel = [5, 5, 5]  # large filter size
    sm_kernel = [2, 2, 2]  # small filter size
    ssm_kernel = [1, 1, 1]  # filter for 1x1x1 conv at end
    lg_stride = [2, 2, 2]  # stride for large filter
    sm_stride = [1, 1, 1]  # stride for small filter

    # Create model input ()
    # * NOTE: samples = channels in paper
    input_vol = Input(shape=(input_size, input_size, input_size, samples))

    ############################
    # Starting level 1 downsampling
    ############################
    conv1 = Conv3D(
        num_filters[0],
        lg_kernel,
        strides=sm_stride,
        padding='same',
        activation='PReLU',
        name='down-conv1_lv1'
    )(input_vol)

    sum1 = Add()([input_vol, conv1])

    down1 = Conv3D(
        num_filters[0],
        sm_kernel,
        strides=lg_stride,
        padding='valid',
        activation='PReLU',
        name='down_lv1'
    )(sum1)

    ############################
    # Starting level 2 downsampling
    ############################
    conv2 = Conv3D(
        num_filters[1],
        lg_kernel,
        strides=sm_stride,
        padding='same',
        activation='PReLU',
        name='down-conv1_lv2'
    )(down1)

    conv2 = Conv3D(
        num_filters[1],
        lg_kernel,
        strides=sm_stride,
        padding='same',
        activation='PReLU',
        name='down-conv2_lv2'
    )(conv2)

    sum2 = Add()([down1, conv2])

    down2 = Conv3D(
        num_filters[1],
        sm_kernel,
        strides=lg_stride,
        padding='valid',
        activation='PReLU',
        name='down_lv2'
    )(sum2)

    ############################
    # Starting level 3 downsampling
    ############################
    conv3 = Conv3D(
        num_filters[2],
        lg_kernel,
        strides=sm_stride,
        padding='same',
        activaiion='PReLU',
        name='down-conv1_lv3'
    )(down2)

    conv3 = Conv3D(
        num_filters[2],
        lg_kernel,
        strides=sm_stride,
        padding='same',
        activaiion='PReLU',
        name='down-conv2_lv3'
    )(conv3)

    conv3 = Conv3D(
        num_filters[2],
        lg_kernel,
        strides=sm_stride,
        padding='same',
        activaiion='PReLU',
        name='down-conv3_lv3'
    )(conv3)

    sum3 = Add()([down2, conv3])

    down3 = Conv3D(
        num_filters[2],
        sm_kernel,
        strides=lg_stride,
        padding='valid',
        activation='PReLU',
        name='down_lv3'
    )(sum3)

    ############################
    # Starting level 4 downsampling
    ############################
    conv4 = Conv3D(
        num_filters[3],
        lg_kernel,
        strides=sm_stride,
        padding='same',
        activaiion='PReLU',
        name='down-conv1_lv4'
    )(down3)

    conv4 = Conv3D(
        num_filters[3],
        lg_kernel,
        strides=sm_stride,
        padding='same',
        activaiion='PReLU',
        name='down-conv2_lv4'
    )(conv4)

    conv4 = Conv3D(
        num_filters[3],
        lg_kernel,
        strides=sm_stride,
        padding='same',
        activaiion='PReLU',
        name='down-conv3_lv4'
    )(conv3)

    sum4 = Add()([down3, conv4])

    down4 = Conv3D(
        num_filters[3],
        sm_kernel,
        strides=lg_stride,
        padding='valid',
        activation='PReLU',
        name='down_lv4'
    )(sum4)

    ############################
    # Starting level 5 (upsample starts here)
    ############################
    conv5 = Conv3D(
        num_filters[4],
        lg_kernel,
        strides=sm_stride,
        padding='same',
        activaiion='PReLU',
        name='conv1_lv5'
    )(down4)

    conv5 = Conv3D(
        num_filters[4],
        lg_kernel,
        strides=sm_stride,
        padding='same',
        activaiion='PReLU',
        name='conv2_lv5'
    )(conv5)

    conv5 = Conv3D(
        num_filters[4],
        lg_kernel,
        strides=sm_stride,
        padding='same',
        activaiion='PReLU',
        name='conv3_lv5'
    )(conv5)

    sum5 = Add()([down4, conv5])

    up1 = Conv3DTranspose(
        num_filters[4],
        sm_kernel,
        strides=lg_stride,
        padding='valid',
        output_padding=None,
        activation='PReLU',
        name='up_lv5'
    )(sum5)

    ############################
    # Starting level 4 upsampling
    ############################

    concat1 = concatenate([up1, sum4], axis=4)
    conv6 = Conv3D(
        num_filters[3],
        lg_kernel,
        strides=sm_stride,
        padding='same',
        activaiion='PReLU',
        name='up-conv1_lv4'
    )(concat1)

    conv6 = Conv3D(
        num_filters[3],
        lg_kernel,
        strides=sm_stride,
        padding='same',
        activaiion='PReLU',
        name='up-conv2_lv4'
    )(conv6)

    conv6 = Conv3D(
        num_filters[3],
        lg_kernel,
        strides=sm_stride,
        padding='same',
        activaiion='PReLU',
        name='up-conv3_lv4'
    )(conv6)

    sum6 = Add()([up1, conv6])

    up2 = Conv3DTranspose(
        num_filters[3],
        sm_kernel,
        strides=lg_stride,
        padding='valid',
        output_padding=None,
        activation='PReLU',
        name='up_lv4'
    )(sum6)

    ############################
    # Starting level 3 upsampling
    ############################

    concat2 = concatenate([up2, sum3], axis=4)
    conv7 = Conv3D(
        num_filters[2],
        lg_kernel,
        strides=sm_stride,
        padding='same',
        activaiion='PReLU',
        name='up-conv1_lv3'
    )(concat2)

    conv7 = Conv3D(
        num_filters[2],
        lg_kernel,
        strides=sm_stride,
        padding='same',
        activaiion='PReLU',
        name='up-conv2_lv3'
    )(conv7)

    conv7 = Conv3D(
        num_filters[2],
        lg_kernel,
        strides=sm_stride,
        padding='same',
        activaiion='PReLU',
        name='up-conv3_lv3'
    )(conv7)

    sum7 = Add()([up2, conv7])

    up3 = Conv3DTranspose(
        num_filters[2],
        sm_kernel,
        strides=lg_stride,
        padding='valid',
        output_padding=None,
        activation='PReLU',
        name='up_lv3'
    )(sum7)

    ############################
    # Starting level 2 upsampling
    ############################

    concat3 = concatenate([up3, sum2], axis=4)
    conv8 = Conv3D(
        num_filters[1],
        lg_kernel,
        strides=sm_stride,
        padding='same',
        activaiion='PReLU',
        name='up-conv1_lv2'
    )(concat3)

    conv8 = Conv3D(
        num_filters[1],
        lg_kernel,
        strides=sm_stride,
        padding='same',
        activaiion='PReLU',
        name='up-conv2_lv2'
    )(conv8)

    sum8 = Add()([up3, conv8])

    up4 = Conv3DTranspose(
        num_filters[1],
        sm_kernel,
        strides=lg_stride,
        padding='valid',
        output_padding=None,
        activation='PReLU',
        name='up_lv2'
    )(sum8)

    ############################
    # Starting level 1 upsampling
    ############################

    concat4 = concatenate([up4, sum1], axis=4)
    conv9 = Conv3D(
        num_filters[0],
        lg_kernel,
        strides=sm_stride,
        padding='same',
        activaiion='PReLU',
        name='up-conv1_lv1'
    )(concat4)

    conv9 = Conv3D(
        num_filters[0],
        lg_kernel,
        strides=sm_stride,
        padding='same',
        activaiion='PReLU',
        name='up-conv2_lv1'
    )(conv9)

    sum9 = Add()([up4, conv9])

    ############################
    # Generating output
    ############################
    conv10 = Conv3D(
        1,
        ssm_kernel,
        strides=sm_stride,
        padding='same',
        name='1b1-conv_lv1'
    )(sum9)

    output = softmax(conv10, axis=-1)

    model = Model(inputs=[input_vol], outputs=[output])

    return model
