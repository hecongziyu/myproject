import data.char as c
from warpctc_pytorch import CTCLoss
import utils as utils
import torch.optim as optim
import os
from model import CRNN
# import importlib
# importlib.reload(utils)


ngpu = 0
# size of the lstm hidden state
nh = 256
nclass = len(c.alphabet) + 1
# input channel ， 因为训练图片是转成灰度图，所以该值为1
nc = 1
lr = 0.001
beta1=0.5
MOMENTUM = 0.9
EPOCH = 100




# 字符转换编码
converter = utils.strLabelConverter(c.alphabet)
# 损失函数
criterion = CTCLoss()

crnn = CRNN(imgH, nc, nclass, nh, ngpu)
crnn.apply(weights_init)
if os.path.exists('/home/hecong/temp/data/ocr/simple_ocr.pkl'):
    crnn.load_state_dict(torch.load('/home/hecong/temp/data/ocr/simple_ocr.pkl'))

image = torch.FloatTensor(batchSize, 3, imgH, imgH)
text = torch.IntTensor(batchSize * 5)
length = torch.IntTensor(batchSize)

# optimizer = optim.Adam(
#     crnn.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer = optim.SGD(
    crnn.parameters(), lr=lr, momentum=MOMENTUM)

for epoch in range(EPOCH):
    for step,(t_image,t_label) in enumerate(train_loader):
        batch_size = t_image.size(0)
        utils.loadData(image, t_image)
        t, l = converter.encode(t_label)
        utils.loadData(text, t)
        utils.loadData(length, l)
        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        optimizer.zero_grad()
        cost = criterion(preds, text, preds_size, length) / batch_size
        cost.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print('{}:{} loss --> {}'.format(epoch, step, cost))
            torch.save(crnn.state_dict(), '/home/hecong/temp/data/ocr/simple_ocr.pkl')