import json
import config
from re_instance_extractor import REInstanceExtractor
from mln_generator import MLNGenerator


def read_input():
    f = open(config.data_path+'input', 'r', encoding='utf-8')
    data_obj_list = []
    for line in f:
        if (len(line) < 2):
            continue
        try:
            data = json.loads(line.strip())
            data_obj_list.append(data['sentence'][0])
        except:
            data_obj_list = []
    f.close()
    return data_obj_list

def main():
    data_obj_list = read_input()

    # input을 읽어서 관계를 추출할 instance들(문장/sbj-obj쌍/Feature) 목록을 생성한다.
    inst_extractor = REInstanceExtractor()
    re_instance_list = []
    for data_obj in data_obj_list:
        result = inst_extractor.extract_re_instance(data_obj)
        re_instance_list.extend(result)

    # instance 정보들을 Markov Logic Network에 들어가는 evidence grounding들로 만든다.
    MLNGenerator().write_mln_data(re_instance_list)

if __name__ == '__main__':
    main()