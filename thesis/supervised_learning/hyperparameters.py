image_size = 50
'''ConvNet'''
k1_size = 6
k1_stride = 1
k1_depth = 1
k1_nm = 16
n1 = image_size*image_size*1

k2_size = 3
k2_stride = 2
k2_depth = 16
k2_nm = 16
m1_size = image_size-k1_size+k1_stride
n2 = m1_size*m1_size*k1_nm

k3_size = 6
k3_stride = 1
k3_depth = 16
k3_nm = 32
m2_size = (m1_size-k2_size)/k2_stride+1
n3 = m2_size*m2_size*k2_nm

k4_size = 3
k4_stride = 2
k4_depth = 32
k4_nm = 32
m3_size = (m2_size-k3_size)/k3_stride+1
n4 = m3_size*m3_size*k3_nm

k5_size = 3
k5_stride = 1
k5_depth = 32
k5_nm = 64
m4_size = (m3_size-k4_size)/k4_stride+1
n5 = m4_size*m4_size*k4_nm

k6_size = 2
k6_stride = 2
k6_depth = 64
k6_nm = 64
m5_size = (m4_size-k5_size)/k5_stride+1
n6 = m5_size*m5_size*k5_nm

'''Class FC'''
f7_class_size = 120
m6_class_size = (m5_size-k6_size)/k6_stride+1
n7_class = m6_class_size*m6_class_size*k6_nm

f8_class_size = 60
n8_class = f7_class_size

classes_size = 11
n9_class = f8_class_size

'''Angle FC'''
f7_angle_size = 120
m6_angle_size = (m5_size-k6_size)/k6_stride+1
n7_angle = m6_angle_size*m6_angle_size*k6_nm

f8_angle_size = 60
n8_angle = f7_angle_size

angles_size = 10
n9_angle = f8_angle_size

'''Dropout'''
keep_prob1 = 0.8
keep_prob2 = 0.5

'''Mini-batch'''
batch_size = 33

