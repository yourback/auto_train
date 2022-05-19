import json
import os
import random
import shutil
import sys
import time


def jsons2txt(src_dir, dst_dir):
    prexs = src_dir.split("/")
    # print(prexs)
    prex = prexs[-1]
    # print(prex)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    for file in os.listdir(src_dir):
        if file.split('.')[-1] == "json":
            print(file)
            # rects = []
            # points = np.zeros((7, 4))
            f = open(os.path.join(src_dir, file))
            jsons = json.load(f)
            datas = jsons["shapes"]
            h = jsons["imageHeight"]
            w = jsons["imageWidth"]
            fw = open(os.path.join(dst_dir, prex + "_" + file.split('.')[0] + ".txt"), 'w')
            for data in datas:
                if data["label"] != "rect2":
                    left = data["points"][0][0]
                    top = data["points"][0][1]
                    right = data["points"][1][0]
                    bottom = data["points"][1][1]
                    x = (left + right) / 2.0 / w
                    y = (top + bottom) / 2.0 / h
                    w = (right - left + 1) / w
                    h = (bottom - top + 1) / h
                    fw.write("0 " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")
            fw.close()
            shutil.copy(os.path.join(src_dir, file.split('.')[0] + ".jpg"),
                        os.path.join(dst_dir, prex + "_" + file.split('.')[0] + ".jpg"))


def gen_train_and_test_with(lst, weight_for_train: float = 0.9):
    with open(lst, "r") as f:
        lines = f.readlines()
    random.shuffle(lines)
    f_train = open(lst + "_train", "w")
    for line in lines[0: round(len(lines) * weight_for_train)]:
        line = line.strip()
        f_train.write(line + '\n')
    f_train.close()
    f_test = open(lst + "_test", "w")
    for line in lines[round(len(lines) * weight_for_train):]:
        line = line.strip()
        f_test.write(line + '\n')
    f_test.close()


class ObjTrain:
    __DARKNET_PATH = "/home/vision/sy_ws/darknet"

    __IS_TINY: bool = True

    __FILE_LIST_PATH = "/home/vision/sy_ws/all_pics.lst"
    __FILE_LIST_TEST_PATH = "%s_test" % __FILE_LIST_PATH
    __FILE_LIST_TRAIN_PATH = "%s_train" % __FILE_LIST_PATH

    __FILE_CFG_PATH = "/home/vision/sy_ws/darknet/cfg"

    def __init__(self):
        self.topic = "temp_obj"
        self.time_tag = '2022'
        self.names_txt = self.topic

        self.SRC_PICS_PATH = "/home/vision/sy_ws/sy_imgs"
        self.DST_PICS_PATH = "/home/vision/sy_ws/sy_imgs_ready"

        self.FILE_NAMES_PATH = "%s/obj_%s.names" % (self.__FILE_CFG_PATH, self.topic)
        self.FILE_DATA_PATH = "%s/obj_%s.data" % (self.__FILE_CFG_PATH, self.topic)
        self.FILE_BACKUP_PATH = "/home/vision/sy_ws/darknet/backup/%s_%s" % (self.time_tag, self.topic)

        self.DATA_TXT = """classes = 1
                train = %s
                valid = %s
                names = %s
                backup = %s
                """ % (
            self.__FILE_LIST_TRAIN_PATH, self.__FILE_LIST_TEST_PATH, self.FILE_NAMES_PATH, self.FILE_BACKUP_PATH)

    def __init_dirs(self):
        if not os.path.exists(self.FILE_BACKUP_PATH):
            os.mkdir(self.FILE_BACKUP_PATH)

    def set_topic(self, topic_str: str):
        """
        设置 标记类名
        :param topic_str:
        :return:self
        """
        self.topic = topic_str
        self.time_tag = self.get_current_time()
        self.names_txt = self.topic
        return self

    def set_src_pics_path(self, path: str):
        """
        设置 图片源
        :param path: 图片源路径
        :return: self
        """
        self.SRC_PICS_PATH = path
        self.DST_PICS_PATH = path + '_ready'
        return self

    def init_params(self):
        self.FILE_NAMES_PATH = "%s/obj_%s.names" % (self.__FILE_CFG_PATH, self.topic)
        self.FILE_DATA_PATH = "%s/obj_%s.data" % (self.__FILE_CFG_PATH, self.topic)
        self.FILE_BACKUP_PATH = "/home/vision/sy_ws/darknet/backup/%s_%s" % (self.time_tag, self.topic)

        self.DATA_TXT = """classes = 1
        train = %s
        valid = %s
        names = %s
        backup = %s
        """ % (self.__FILE_LIST_TRAIN_PATH, self.__FILE_LIST_TEST_PATH, self.FILE_NAMES_PATH, self.FILE_BACKUP_PATH)
        return self

    def __generate_txt(self):
        # /home/vision/sy_ws/darknet/data
        jsons2txt(self.SRC_PICS_PATH, self.DST_PICS_PATH)

    def __generate_train_test_list(self):
        os.system("ls %s/*jpg>%s" % (self.DST_PICS_PATH, self.__FILE_LIST_PATH))
        gen_train_and_test_with(self.__FILE_LIST_PATH)

    def __generate_file_names(self):
        with open(self.FILE_NAMES_PATH, 'w') as f:
            f.write(self.names_txt)

    def __generate_file_data(self):
        with open(self.FILE_DATA_PATH, 'w') as f:
            f.write(self.DATA_TXT)

    def start_train(self):
        self.__init_train_data()
        # ./darknet   detector   train    obj.data    cfg/yolov4-custom.cfg    ./yolov4.conv.137
        c = "./darknet detector train %s %s %s" % (
            self.FILE_DATA_PATH,
            "cfg/yolov4-tiny-custom.cfg" if self.__IS_TINY else "cfg/yolov4-custom.cfg",
            "./yolov4.conv.137" if self.__IS_TINY else "./yolov4-tiny.conv.29")
        command = """gnome-terminal  -e 'bash -c  \"cd %s && %s ;exec bash\"'""" % (
            self.__DARKNET_PATH, c)
        os.system(command)

    def continue_train(self):
        c = "./darknet detector train %s %s %s" % (
            self.FILE_DATA_PATH,
            "cfg/yolov4-tiny-custom.cfg" if self.__IS_TINY else "cfg/yolov4-custom.cfg",
            "%s/%s" % (
                self.FILE_BACKUP_PATH,
                'yolov4-tiny-custom_last.weights' if self.__IS_TINY else 'yolov4-custom_last.weights'))

        command = """gnome-terminal  -e 'bash -c  \"cd %s && %s ;exec bash\"'""" % (
            self.__DARKNET_PATH, c)
        os.system(command)

    def __init_train_data(self):
        self.__init_dirs()
        # 生成 json文件

        # 生成txt文件
        self.__generate_txt()

        # 生成两个列表
        self.__generate_train_test_list()

        # 生成obj.names
        self.__generate_file_names()

        # 生成obj.data
        # classes = 1
        # train = train_list
        # valid = test_list
        # names = obj.names
        # backup = backup / 0105 /
        self.__generate_file_data()

    @staticmethod
    def get_current_time():
        return time.strftime("%y%m%d", time.localtime())

    def set_tiny(self, tiny: str):
        self.__IS_TINY = tiny.startswith('t')
        return self


if __name__ == '__main__':
    s = len(sys.argv)
    if s == 3:
        print("继续训练")
        topic = sys.argv[1]
        is_tiny = sys.argv[2]
        ObjTrain().set_topic(topic).init_params().set_tiny(is_tiny).continue_train()
    else:
        print("重新开始训练")
        topic = sys.argv[1]
        is_tiny = sys.argv[2]
        src_pics_path = sys.argv[3]
        ObjTrain().set_topic(topic).set_src_pics_path(src_pics_path).set_tiny(is_tiny).init_params().start_train()
