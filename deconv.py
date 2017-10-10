import tensorflow as tf

x1 = tf.constant(1.0, shape=[1,3,3,1])      # 3x3, 1通道
x2 = tf.constant(1.0, shape=[1,6,6,3])      # 6x6, 3通道
x3 = tf.constant(1.0, shape=[1,5,5,3])      # 5x5, 3通道
kernel = tf.constant(1.0, shape=[3,3,3,1])  #[filter_height, filter_width, in_channels, out_channels]
y2 = tf.nn.conv2d(x3, kernel, strides=[1,2,2,1], padding="SAME")  
y3 = tf.nn.conv2d_transpose(y2,kernel,output_shape=[1,5,5,3], strides=[1,2,2,1],padding="SAME")  