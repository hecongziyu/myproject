import lib.data.char as c
import lib.utils as utils
import os
import importlib
importlib.reload(utils)


ngpu = 0
# size of the lstm hidden state
nh = 256
nclass = len(c.alphabet) + 1
# input channel ， 因为训练图片是转成灰度图，所以该值为1
nc = 1
lr = 0.001
beta1=0.5
converter = utils.strLabelConverter(c.alphabet)

crnn = CRNN(imgH, nc, nclass, nh, ngpu)
crnn.apply(weights_init)
if os.path.exists('/home/hecong/temp/data/ocr/simple_ocr.pkl'):
    crnn.load_state_dict(torch.load('/home/hecong/temp/data/ocr/simple_ocr.pkl'))
image = torch.FloatTensor(batchSize, 3, imgH, imgH)
_,(v_image,v_text) = next(enumerate(train_loader))
utils.loadData(image,v_image)
preds_s = crnn(image)
batch_size = v_image.size(0)
preds_size = Variable(torch.IntTensor([preds_s.size(0)] * batch_size))

preds = preds_s.clone()

preds = preds.permute(1,0,2)
print(preds.size())
_,preds = preds.max(2)
preds = preds.view(-1)
# print(preds)
preds_size = Variable(torch.IntTensor([preds_s.size(0)])) * batchSize
sim_preds = converter.decode(preds.data, preds_size.data, raw=False)

print(sim_preds)
image_0 = v_image[1][0]
image_0 = image_0.numpy()
plt.imshow(image_0,'gray')
plt.show()
