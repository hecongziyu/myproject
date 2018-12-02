# -*- coding: utf-8 -*-

# https://www.cnblogs.com/jerrybaby/p/5849092.html

# -*- coding: utf-8 -*-

import cv2
import numpy as np
import fire
# 当鼠标按下时变为True
drawing=False
# 如果mode为true绘制矩形。按下'm' 变成绘制曲线。
mode=True
ix,iy=-1,-1

file_path = 'd:\\'
file_name = 'cc_10.jpg'
save_name = 'cc_10.txt'

src = cv2.imread('{}{}'.format(file_path,file_name))
img = src.copy()

boxes = []

# 创建回调函数
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    # 当按下左键是返回起始位置坐标
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        print('L --> {} : {}  R --> {} : {}'.format(ix,iy,x,y))
        boxes.append([ix,iy,ix,y,x,y,x,iy])
        
    # 当鼠标左键按下并移动是绘制图形。event可以查看移动，flag查看是否按下
    elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
        if drawing==True:
            if mode==True:
                # print('L --> {} : {}  R --> {} : {}'.format(ix,iy,x,y))
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)
                # cv2.rectangle(img,(384,0),(510,128),(0,255,0),3) 不填充
            else:
                # 绘制圆圈，小圆点连在一起就成了线,3代表了笔画的粗细
                cv2.circle(img,(x,y),3,(0,0,255),-1)
                # 下面注释掉的代码是起始点为圆心，起点到终点为半径的
                # r=int(np.sqrt((x-ix)**2+(y-iy)**2))
                # cv2.circle(img,(x,y),r,(0,0,255),-1)
        # 当鼠标松开停止绘画。
        # elif event==cv2.EVENT_LBUTTONUP:
        #     print('L --> {} : {}  R --> {} : {}'.format(ix,iy,x,y))
        #     drawing==False
        #     # if mode==True:
        #         # cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
        #     # else:
        #         # cv2.circle(img,(x,y),5,(0,0,255),-1)
		
def split_box_size(width=16):
    out_boxes = []
    # print(in_boxes)
    print(boxes)
    for box in boxes:
        # box begin_x, begin_y, end_x, end_y
        box = np.array(box)
        print('box --> {}'.format(box))
        box = box.reshape(-1,2)

        box = box[box[:,0].argsort()]
        start_box  = box[:2]
        end_box = box[2:]
        number = int((end_box[0][0] - start_box[0][0])/width)

        # import ipdb
        # ipdb.set_trace()
        # 按最后一列降序
        start_box=start_box[np.lexsort(-start_box.T)]
        old_box = start_box
        for i in np.arange(1,(1+number),1):
            tmp_box = start_box + ([width*i,0])
            tmp_box = tmp_box[tmp_box[:,1].argsort()]
            new_box = np.concatenate((old_box,tmp_box))
            old_box = tmp_box[np.lexsort(-tmp_box.T)]
            out_boxes.append(new_box.reshape(-1).tolist())

    return out_boxes
	
		
def save_boxes():
    print('save boxes begin')
    s_boxes = split_box_size()
    with open('{}{}'.format(file_path,save_name), 'w+') as fw:
        for item in s_boxes:
            fw.write(",".join(map(str,item))+"\n")

    print('save boxes over')

def show_img_box(in_boxes):
    img = src.copy()
    region = []
    for item in in_boxes:
        box = np.array(item)
        box = box.reshape(4,-1)
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        if (height > width * 1.3):
            continue
        region.append(box)
    for box in region:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 1)

    cv2.imshow("img",img)
    cv2.waitKey(0)


def mouse_click():
    # 回调函数与OpenCV 窗口绑定在一起,
    # 在主循环中我们需要将键盘上的“m”键与模式转换绑定在一起。
    cv2.namedWindow('image')
    # 绑定事件
    cv2.setMouseCallback('image',draw_circle) 
    while(1):
        cv2.imshow('image',img)
        k=cv2.waitKey(1)&0xFF
        if k==ord('m'):
            mode=not mode
        elif k == ord('s'):
            save_boxes()
            show_img_box(boxes)
            break
        elif k==27:
            break
            
if __name__ == '__main__':
    fire.Fire()
            
            