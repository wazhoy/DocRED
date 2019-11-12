import json
import os

def count_line():
    ori_filepath = "/home/wzy/PycharmProjects/DocRED/data/train_annotated.json"
    ori_data = json.load(open(ori_filepath))
    split_line = list(range(0,len(ori_data),int(len(ori_data)/5)+1))
    data_slices = list()
    for i in range(4):
        data_slices.append(ori_data[split_line[i]:split_line[i+1]])
    data_slices.append(ori_data[split_line[4]:])

    total_len = 0
    for _slice  in data_slices:
        total_len += len(_slice)
    assert total_len == len(ori_data)
    assert len(ori_data) == 3053
    assert len(data_slices) == 5

    for cur in range(5):
        dev_slice = data_slices[cur]
        train_slice = list()
        for _cur in range(5):
            if _cur != cur:
                train_slice.extend(data_slices[_cur])
        if cur != 4:
            assert len(dev_slice) == 611
            assert len(train_slice) == 3053 - 611
        if cur == 4:
            assert len(dev_slice) == 3053 - 4 * 611
            assert len(train_slice) == 4 * 611
        dir_path = "/home/wzy/PycharmProjects/DocRED/"+str(cur+1)+"of5part_data/"
        json.dump(train_slice, open(os.path.join(dir_path, 'train_annotated.json'), "w"))
        json.dump(dev_slice, open(os.path.join(dir_path, 'dev_annotated.json'), "w"))
if __name__ == "__main__":
    count_line()