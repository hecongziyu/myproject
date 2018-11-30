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

src = cv2.imread('d:\\5.jpg')
img = src.copy()
# img=np.zeros((512,512,3),np.uint8)
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
        elif k==27:
            break
            
if __name__ == '__main__':
    fire.Fire()
            
            