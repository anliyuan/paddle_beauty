import paddle
import cv2
import numpy as np
from model.model import Net

img = cv2.imread('./0.jpg')
img_ori = img.copy()
h, w = img_ori.shape[:2]
img = cv2.resize(img, (320, 320))
img = img.transpose((2,0,1))

img = np.float32(img)/255.0
img = np.expand_dims(img, axis=0)
img = paddle.to_tensor(img, dtype='float32')


p = './checkpoint/0.pdparams'
net = Net(3)
net.set_dict(paddle.load(p))
pred = net(img)
pred = pred.squeeze()
pred = pred.cpu()#.data.numpy().transpose((1,2,0))
pred_np = np.asarray(pred).transpose((1,2,0))
pred_np = (pred_np*2.55)
pred_np = cv2.resize(pred_np, (w, h))
img_ori = np.float32(img_ori)
res = img_ori + pred_np
res[res>255]=255
res[res<0]=0
res = np.uint8(res)
# cv2.imshow('img', res)
# cv2.waitKey(0)
cv2.imwrite('./res.jpg', res)
