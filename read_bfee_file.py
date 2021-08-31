from Bfee import Bfee
from CSI.gesture_recognition.get_scale_csi import get_scale_csi

if __name__ == '__main__':
    filename = 'E:/project/CSI/classroom/DX/O/1/log.dat'
    bfee = Bfee.from_file(filename, model_name_encode="gb2312")
    print(len(bfee.dicts))
    print(len(bfee.all_csi))
    for i in range(len(bfee.all_csi)):
        csi = get_scale_csi(bfee.dicts[i])
        print(csi)
        print("*************",i)

