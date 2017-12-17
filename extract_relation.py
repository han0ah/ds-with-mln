import json
import config
import subprocess
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


def extract_re_instances(data_obj_list):
    # input을 읽어서 관계를 추출할 instance들(문장/sbj-obj쌍/Feature) 목록을 생성한다.
    inst_extractor = REInstanceExtractor()
    re_instance_list = []
    for data_obj in data_obj_list:
        result = inst_extractor.extract_re_instance(data_obj)
        re_instance_list.extend(result)
    return re_instance_list

def write_markov_logic_network_data(re_instance_list):
    # instance 정보들을 Markov Logic Network에 들어가는 evidence grounding들로 만든다.
    MLNGenerator().write_mln_data(re_instance_list)

def run_alchemy_inference():
    # Alchemy를 통해 Markov Logic Network Inference를 한다.
    'infer -ms -i re-learnt.mln -r re_train.result -e train.db -q Label,HasRel'
    bashCommand = "{} -ms -i {} -r {} -e {} -q Label,HasRel".format(config.alchemy_path+'infer',
                                                                    config.data_path+'re-learnt.mln',
                                                                    config.data_path+'re_test.result',
                                                                    config.data_path+'test.db')
    result = subprocess.call(bashCommand.split())
    print (result)

def main():
    data_obj_list = read_input()
    re_instance_list = extract_re_instances(data_obj_list)
    write_markov_logic_network_data(re_instance_list)
    run_alchemy_inference()

if __name__ == '__main__':
    main()