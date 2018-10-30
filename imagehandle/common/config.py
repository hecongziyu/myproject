# -*- coding: utf-8 -*-
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.BASE = edict()
__C.BASE.QS_TYPE_SINGLE_CHOICE = 1     # 单选择题
__C.BASE.QS_TYPE_MUTI_CHOICE = 2     # 多选择题
__C.BASE.QS_TYPE_JUDGE = 3      #   判断题
__C.BASE.QS_TYPE_FILL = 4      # 填空题
__C.BASE.QS_TYPE_NORMAL = 5     # 问答题
__C.BASE.QS_TYPE_B = 6     # 大小题
# 需特殊处理手机型号
__C.BASE.SPECIAL_MOBILE = []
__C.BASE.DEBUG = False

# 文件服务，用于保存图片
__C.FILE = edict()
__C.FILE.client_id = '463af366-f61e-4052-8cad-b32d1819bfea'
__C.FILE.client_secret = 'c848f970-cd58-43e9-8962-4e2fc6f2eae7'
__C.FILE.GET_TOKEN_URL = 'http://test.openapi.cslearning.cn/basefileApi/fileApi/getFileToken'
__C.FILE.UPLOD_FILE_URL = 'http://test.openapi.cslearning.cn/zuul/basefileApi/fileUpload/normal'

__C.OCR = edict()
__C.OCR.USER_NAME = 'talkweb_edu1'
__C.OCR.PASSWORD = 'talkweb@123456'
__C.OCR.AUTH_URL = 'https://iam.cn-north-1.myhuaweicloud.com/v3/auth/tokens'
__C.OCR.OCR_URL = 'https://ais.cn-north-1.myhuaweicloud.com/v1.0/ocr/general-text'
__C.OCR.SCOPE = 'cn-north-1'




# 试卷图片分割基本参数, 针对大图片
__C.IMAGE = edict()
# 试卷图片生成基础路径
__C.IMAGE.BASE_PATH = './static/images/'
# 试卷图片生成后URL访问路径
__C.IMAGE.BASE_URL = '/static/images/'
# 用需要文字识别， 设置图片最小的高、宽，并且比例需为 3 : 4
__C.IMAGE.MIN_WIDTH = 1860
__C.IMAGE.MIN_HIGH = 2480
__C.IMAGE.WIDTH_HIGH_SCALE = 0.75
# 图片模糊度阀值
__C.IMAGE.BLUR_THRESHOLD = 30

# 试卷图片模糊框大小， 分别针对不同大小的图片, 大的图片用大的模糊框
__C.IMAGE.BLUR_RECT = (5, 5)
__C.IMAGE.LINE_DILATE_RECT = (21, 21)
# 试卷图片直线长度设置，用于扫描直线
__C.IMAGE.MIN_LINE_LENGTH = 1000
__C.IMAGE.MAX_LINE_GAP = 80
# 最小倾斜角度, 试卷超过该角度则不通过
__C.IMAGE.MIN_LEAN_ANGLE = 10
# 可纠正倾斜角度, 试卷超过该角度则进行纠正
__C.IMAGE.CORRECT_LEAN_ANGLE = 0.1

# 试卷边宽的宽和高的长度最小设置
__C.IMAGE.MIN_LINE_WIDTH = 1200
__C.IMAGE.MIN_LINE_HEIGHT = 1000
__C.IMAGE.MIN_HEIGHT_INTERVAL = 200
__C.IMAGE.MIN_WIDTH_INTERVAL = 100

# 字体
__C.IMAGE.FONT_NAME= 'HYChunRanShouShuW.ttf'


# 消息返回定义
__C.RET = edict()
__C.RET.SUCCESS = (0,'处理成功')
__C.RET.ERROR_BLUR = (11, '图片太模糊')
__C.RET.ERROR_WIDTH_HIGH_SCALE = (12, '图片高度宽度比不为4比3')
__C.RET.ERROR_EXAMPLE_ID_EXIST = (13,'试卷ID已存在')
__C.RET.ERROR_LEAN_ANGLE = (14,'试卷倾斜角度过大，需重新拍照')
__C.RET.ERROR_HIGH = (15,'试卷高度太低，会影响文字识别精度，请重新拍照')

__C.RET.ERROR_HW_OCR = (21, '华为OCR接口请求太多，请稍后再试')
__C.RET.ERROR_HW_AUTH = (22, '华为OCR接口用户认证失败')
__C.RET.ERROR_HW_FAILUE = (23, '华为OCR接口出现异常，需查找具体原因')
__C.RET.ERROR_EXAM_QUESTION_NUM = (31, '试卷题目数与上传图片分隔的题数不相同')
__C.RET.ERROR_FILE_HTTP_STATUS = (41, '文件上传服务取HTTP返回状态码不为200')
__C.RET.ERROR_FILE_TOKEN_AUTH = (42, '文件上传服务取TOKEN失败，返回包体代码不为200')
__C.RET.ERROR_FILE_UPLOAD = (43, '文件上传服务取TOKEN失败，返回包体代码不为200')


__C.RET.ERROR_OTHER = (99, '系统出现异常')







