import numpy as np

class Bfee:

    def __init__(self):
        pass

    @staticmethod
    def from_file(filename, model_name_encode="shift-JIS"):

        with open(filename, "rb") as f:
            from functools import reduce
            array = bytes(reduce(lambda x, y: x+y, list(f))) # reduce(函数，list)，将list中元素依次累加

        bfee = Bfee()
        
#         vmd.current_index = 0
        bfee.file_len = len(array)
        bfee.dicts = []
        bfee.all_csi = []

#         vmd.timestamp_low0 = int.from_bytes(array[3:7], byteorder='little', signed=False)
        
#         array = array[3:]   
        
        #%% Initialize variables
        #ret = cell(ceil(len/95),1);    # % Holds the return values - 1x1 CSI is 95 bytes big, so this should be upper bound
        cur = 0                       # % Current offset into file
        count = 0                    # % Number of records output
        broken_perm = 0              # % Flag marking whether we've encountered a broken CSI yet
        triangle = [0, 1, 3]           # % What perm should sum to for 1,2,3 antennas
                    
        while cur < (bfee.file_len - 3):
            #% Read size and code
            #% 将文件数据读取到维度为 sizeA 的数组 A 中，并将文件指针定位到最后读取的值之后。fread 按列顺序填充 A。
            bfee.field_len = int.from_bytes(array[cur:cur+2], byteorder='big', signed=False)
            bfee.code = array[cur+2]
            cur = cur+3
            
            # there is CSI in field if code == 187，If unhandled code skip (seek over) the record and continue
            if bfee.code == 187:
                pass
            else:
                #% skip all other info
                cur = cur + bfee.field_len - 1
                continue
            
            # get beamforming or phy data
            if bfee.code == 187:
                count =count + 1
                
                bfee.timestamp_low = int.from_bytes(array[cur:cur+4], byteorder='little', signed=False)
                bfee.bfee_count = int.from_bytes(array[cur+4:cur+6], byteorder='little', signed=False)
                bfee.Nrx = array[cur+8]
                bfee.Ntx = array[cur+9]
                bfee.rssi_a = array[cur+10]
                bfee.rssi_b = array[cur+11]
                bfee.rssi_c = array[cur+12]
                bfee.noise = array[cur+13] - 256
                bfee.agc = array[cur+14]
                bfee.antenna_sel = array[cur+15]
                bfee.len = int.from_bytes(array[cur+16:cur+18], byteorder='little', signed=False)
                bfee.fake_rate_n_flags = int.from_bytes(array[cur+18:cur+20], byteorder='little', signed=False)
                bfee.calc_len = (30 * (bfee.Nrx * bfee.Ntx * 8 * 2 + 3) + 6) / 8
                bfee.csi = np.zeros(shape=(30,bfee.Nrx ,bfee.Ntx), dtype=np.dtype(np.complex))
                bfee.perm = [1,2,3]
                bfee.perm[0] = ((bfee.antenna_sel) & 0x3)
                bfee.perm[1] = ((bfee.antenna_sel >> 2) & 0x3)
                bfee.perm[2] = ((bfee.antenna_sel >> 4) & 0x3)
        

                cur = cur + 20
                
                #get payload
                payload = array[cur:cur+bfee.len]
                cur = cur + bfee.len
                
                index = 0

                #Check that length matches what it should
                if (bfee.len != bfee.calc_len):
                    print("MIMOToolbox:read_bfee_new:size","Wrong beamforming matrix size.")

                #Compute CSI from all this crap :
                #import struct
                for i in range(30):
                    index += 3
                    remainder = index % 8
                    for j in range(bfee.Nrx):
                        for k in range(bfee.Ntx):
                            real_bin = bytes([(payload[int(index / 8)] >> remainder) | (payload[int(index/8+1)] << (8-remainder)) & 0b11111111])
                            real = int.from_bytes(real_bin, byteorder='little', signed=True)
                            imag_bin = bytes([(payload[int(index / 8+1)] >> remainder) | (payload[int(index/8+2)] << (8-remainder)) & 0b11111111])
                            imag = int.from_bytes(imag_bin, byteorder='little', signed=True)
                            tmp = np.complex(float(real), float(imag))
                            bfee.csi[i, j, k] = tmp
                            index += 16
                
                # % matrix does not contain default values
                if sum(bfee.perm) != triangle[bfee.Nrx-1]:
                    print('WARN ONCE: Found CSI (',filename,') with Nrx=', bfee.Nrx,' and invalid perm=[',bfee.perm,']\n' )
                else:
                    temp_csi = np.zeros(bfee.csi.shape, dtype=np.dtype(np.complex))
                    # bfee.csi[:,bfee.perm[0:bfee.Nrx],:] = bfee.csi[:,0:bfee.Nrx,:]
                    for r in range(bfee.Nrx):
                        temp_csi[:,bfee.perm[r],:] = bfee.csi[:,r,:]
                    bfee.csi = temp_csi
                
                # print phy data
#                 print(vmd.file_len,
#                       vmd.field_len,
#                       vmd.code,
#                       vmd.timestamp_low,
#                       vmd.bfee_count,
#                       vmd.Nrx,
#                       vmd.Ntx,
#                       vmd.rssi_a,
#                       vmd.rssi_b,
#                       vmd.rssi_c,
#                       vmd.noise,
#                       vmd.agc,
#                       vmd.antenna_sel,
#                       vmd.len,
#                       vmd.fake_rate_n_flags,
#                       vmd.calc_len,
#                       vmd.perm,
#                       vmd.csi.shape
#                      )

                # 将类属性导出为dict，并返回
                bfee_dict = {}
                bfee_dict['timestamp_low'] = bfee.timestamp_low
                bfee_dict['bfee_count'] = bfee.bfee_count
                bfee_dict['Nrx'] = bfee.Nrx
                bfee_dict['Ntx'] = bfee.Ntx
                bfee_dict['rssi_a'] = bfee.rssi_a
                bfee_dict['rssi_b'] = bfee.rssi_b
                bfee_dict['rssi_c'] = bfee.rssi_c
                bfee_dict['noise'] = bfee.noise
                bfee_dict['agc'] = bfee.agc
                bfee_dict['antenna_sel'] = bfee.antenna_sel
                bfee_dict['perm'] = bfee.perm
                bfee_dict['len'] = bfee.len
                bfee_dict['fake_rate_n_flags'] = bfee.fake_rate_n_flags
                bfee_dict['calc_len'] = bfee.calc_len
                bfee_dict['csi'] = bfee.csi

                bfee.dicts.append(bfee_dict)
                bfee.all_csi.append(bfee.csi)
        
        return bfee

if __name__ == '__main__':
    bfee = Bfee.from_file("E:/project/CSI/classroom/DX/O/1/log.dat", model_name_encode="gb2312")
    print(len(bfee.dicts))
    print(len(bfee.all_csi))
    #print(bfee.all_csi)
    print(bfee.dicts[9])
