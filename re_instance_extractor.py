import config
import data_util
import re

class REInstanceExtractor():

    def extract_re_instance_for_experiment(self, file_name):
        re_instance_list = []
        feature_extractor = FeatureExtractor()
        done_count = 0
        f = open(file_name, 'r', encoding='utf-8')
        for line in f:
            if (len(line) < 2):
                continue
            sbj,obj,relation,template_sent = line.split('\t')
            prev_sbj_loc = template_sent.find('<< _sbj')
            prev_obj_loc = template_sent.find('<< _obj')
            sent = template_sent.replace(' << _sbj_ >> ', sbj)
            sent = sent.replace(' << _obj_ >> ', obj)
            sent = sent.strip()
            template_sent = template_sent.strip()

            sbj_loc_list = [m.start() for m in re.finditer(sbj, sent)]
            obj_loc_list = [m.start() for m in re.finditer(obj, sent)]

            sbj_loc = -1
            obj_loc = -1
            min_diff = 10000
            for loc in sbj_loc_list:
                diff = abs(loc - prev_sbj_loc)
                if (diff < min_diff):
                    sbj_loc = loc
                    min_diff = diff

            min_diff = 10000
            for loc in obj_loc_list:
                diff = abs(loc - prev_obj_loc)
                if (diff < min_diff):
                    obj_loc = loc
                    min_diff = diff

            byte_count = 0
            char_count = 0
            sbj_byte_loc = obj_byte_loc = 0
            for char in sent:
                if (char_count == sbj_loc):
                    sbj_byte_loc = byte_count
                if (char_count == obj_loc):
                    obj_byte_loc = byte_count

                char_count += 1
                byte_count += data_util.get_text_length_in_byte(char)

            nlp_result = data_util.get_nlp_parse_result(sent)
            if (nlp_result != None):
                nlp_result = nlp_result['sentence'][0]

            re_instance = feature_extractor.getFeature(sent,sbj,obj,sbj_byte_loc,obj_byte_loc,nlp_result)
            re_instance['relation'] = relation.strip()
            re_instance_list.append(re_instance)
            done_count += 1
            if (done_count % 50 == 0):
                print('%d data parsing finished'%(done_count))
        f.close()
        return re_instance_list


    def extract_re_instance(self,data_obj):
        '''
        입력 데이터로 부터 Sbj-Obj Entity Pair / Sentence / Feature 값을 가지는 Instance를 뽑아낸다. 
        '''
        # Data 정제
        data_obj = self._add_place_holder_entity(data_obj)
        data_obj = self._revise_entity_index(data_obj)
        data_obj = self._revise_etri_morp_index(data_obj)

        # 주어 sbj-obj 쌍별로 instance Feature 추출
        re_instance_list = []
        feature_extractor = FeatureExtractor()
        num_entity = len(data_obj['entities'])

        if ( config.entity_pair_select_option == "ALL" ):
            i_range = [i for i in range(num_entity)]
        else:
            i_range = []
            sbj_entity_num = self._get_sbj_entiy_num(data_obj)
            if (sbj_entity_num >= 0):
                i_range = [sbj_entity_num]
        for i in i_range:
            for j in range(num_entity):
                if i==j:
                    continue

                sent = data_obj['text']
                sbj = data_obj['entities'][i]['surface_form']
                obj = data_obj['entities'][j]['surface_form']
                sbj_byte_loc = data_obj['entities'][i]['start_offset']
                obj_byte_loc = data_obj['entities'][j]['start_offset']
                re_instance = feature_extractor.getFeature(sent, sbj, obj, sbj_byte_loc, obj_byte_loc, data_obj)
                re_instance_list.append(re_instance)

        return re_instance_list

    def _add_place_holder_entity(self,data_obj):
        '''
        가주어가 있었으면 가주어에 대한 Entity를 추가한다.
        '''
        if(('isAdd' not in data_obj) or (not data_obj['isAdd'])):
            return data_obj
        start_offset = 0
        end_offset = data_obj['ori_text'].index('은')
        uri = 'uri/' + data_obj['text'][0:data_obj['text'].index('은')]
        text = data_obj['ori_text'][0:data_obj['ori_text'].index('은')]

        data_obj['entities'].insert(0, {'start_offset':start_offset,
                                        'end_offset':end_offset,
                                        'text':text,
                                        'uri':uri})
        return data_obj

    def _revise_entity_index(self,data_obj):
        '''
        entity index를 표면형으로 수정된 텍스트기준으로 etri가 사용하는 byte_index로 수정
        '''
        char_offset_to_byte_offset = [0]
        byte_index = 0
        for char in data_obj['text']:
            byte_index += data_util.get_text_length_in_byte(char)
            char_offset_to_byte_offset.append(byte_index)

        sent_index = 0
        entities = data_obj['entities']
        for i in range(len(entities)):
            entity = entities[i]
            entity['surface_form'] = entity['uri'].split('/')[-1]
            entity_length = len(entity['surface_form'])
            while True:
                if (data_obj['text'][sent_index:sent_index+entity_length] == entity['surface_form']):
                    entity['start_offset'] = char_offset_to_byte_offset[ sent_index ]
                    entity['end_offset'] = char_offset_to_byte_offset[ sent_index+entity_length ]
                    break
                sent_index += 1
            entities[i] = entity

        data_obj['entities'] = entities
        return data_obj

    def _revise_etri_morp_index(self, data_obj):
        '''
        ETRI morp index 0부터 시작하도록 수정
        '''
        mval = data_obj['morp'][0]['position']
        for i in range(len(data_obj['morp'])):
            data_obj['morp'][i]['position'] -= mval
        return data_obj

    def _get_sbj_entiy_num(self,data_obj):
        '''
        주어인 entity 번호를 찾는다.
        '''
        for depitem in data_obj['dependency']:
            if (depitem['label'] == 'NP_SBJ'):
                st_idx = data_obj['word'][int(depitem['id'])]['begin']
                st_byte_loc = data_obj['morp'][st_idx]['position']

                entity_num = 0
                for entity in data_obj['entities']:
                    if (entity['start_offset'] == st_byte_loc):
                        return entity_num
                    entity_num += 1
        return -1


class FeatureExtractor():
    def _is_valid_morp(self, morp_tag):
        if (morp_tag.startswith('N') or morp_tag == 'VV' or morp_tag == 'VA'):
            return True
        return False

    def _get_morp_items(self, i, nlp_result, use_vcp=False):
        morp_list = []
        for morp_item in nlp_result['morp_eval'][i]['result'].split('+'):
            items = morp_item.split('/')
            if (len(items) != 2):
                continue
            lemma = items[0]
            morp_type = items[1]
            if (self._is_valid_morp(morp_type)):
                morp_list.append(lemma + '-@-' + morp_type)
            elif (morp_type == 'VCP' and use_vcp and len(morp_list) > 0):
                morp_list[-1] += ('이다-@-VCP')

        return morp_list

    def _removeParenthesis(self, sent):
        remove_keyword = []
        open_count = 0
        st = 0
        new_sent = sent
        index = 0
        for char in new_sent:
            if (char == '('):
                open_count += 1
                if open_count == 1:
                    st = index
            elif (char == ')'):
                open_count -= 1
                if open_count == 0:
                    substr = sent[st:index + 1]
                    if (not (('<< _sbj_ >>' in substr) or ('<< _obj_ >>' in substr))):
                        if (st > 0 and sent[st - 1:st] == ' '):
                            substr = ' ' + substr
                        remove_keyword.append(substr)
            index += 1
        for item in remove_keyword:
            sent = sent.replace(item, '')

        if ('<< _sbj_ >> ' not in sent):
            sent = sent.replace('<< _sbj_ >>', '<< _sbj_ >> ')
        if ('<< _obj_ >> ' not in sent):
            sent = sent.replace('<< _obj_ >>', '<< _obj_ >> ')
        return sent

    ''' Mintz et al(2009)에 나온 방식으로 Relation Extraction을 위한 문장의 feature를 추출한다. '''
    def getFeature(self, sent, sbj,obj, sbj_byte_loc, obj_byte_loc, etri_result=None):
        dummy_result =  {
            'sent' : sent,
            'origin_sent':sent,
            'template_sent':sent,
            'sbj' : sbj,
            'obj' : obj,
            'sbj_ne' : 'NONE',
            'obj_ne' : 'NONE',
            'sbj_label': '',
            'obj_label': '',
            'sbj_josa' : '',
            'obj_josa' : '',
            'dependency': [],
            'dependency_morp': [],
            'arg1_mod': [],
            'arg2_mod': [],
            'context_lemma': []
        }
        origin_sent = sent
        sent = self._removeParenthesis(sent)
        new_template_sent = sent
        sent = sent.replace(' << _sbj_ >> ', sbj)
        sent = sent.replace(' << _obj_ >> ', obj)

        nlp_result = etri_result
        if nlp_result == None:
            return dummy_result

        sbj_id = obj_id = -1
        morp = nlp_result['morp']
        for word in nlp_result['word']:
            word_st = morp[word['begin']]['position']
            word_ed = morp[word['end'] + 1]['position'] - 1 if word['end'] + 1 < len(morp) else 10000
            if sbj_byte_loc >= word_st and sbj_byte_loc <= word_ed:
                sbj_id = word['id']
            if obj_byte_loc >= word_st and obj_byte_loc <= word_ed:
                obj_id = word['id']
        if (sbj_id == -1 or obj_id == -1):
            return dummy_result

        dependency = nlp_result['dependency']
        N = len(dependency)
        graph = [[0 for _ in range(N)] for _ in range(N)]
        graph_label = [['' for _ in range(N)] for _ in range(N)]
        for item in dependency:
            if (item['head'] == -1):
                continue
            graph[item['id']][item['head']] = 1
            graph[item['head']][item['id']] = 2

            graph_label[item['id']][item['head']] = item['label']
            graph_label[item['head']][item['id']] = item['label']

        check_visit = [0 for _ in range(N)]
        queue = [(sbj_id, -1)]
        check_visit[sbj_id] = 1
        now = en = 0
        obj_found = False
        while (now <= en):
            curr_id = queue[now][0]
            for i in range(N):
                if (graph[curr_id][i] > 0 and check_visit[i] == 0):
                    queue.append((i, now))
                    check_visit[i] = 1
                    en += 1
                    if (i == obj_id):
                        obj_found = True
                        break
            if (obj_found):
                break
            now += 1
        now = en
        dependency_path = []
        path = [queue[now][0]]
        while True:
            trace = queue[now][1]
            prev_id = queue[trace][0]
            path.append(prev_id)
            if (prev_id == sbj_id):
                break
            now = trace
        path.reverse()

        for item in dependency:
            item['state_label'] = ''
        for i in path:
            dependency[i]['state_label'] = 'dependency'

        dependency[path[0]]['state_label'] = 'entity'
        dependency[path[-1]]['state_label'] = 'entity'

        arg1_mod = []
        arg2_mod = []

        for i in reversed(range(0, path[0])):
            if ((not dependency[i]['label'].endswith('CNJ')) and dependency[i]['state_label'] != 'entity' and
                        dependency[i]['head'] == path[0]):
                arg1_mod = self._get_morp_items(i, nlp_result)
                dependency[i]['state_label'] = 'arg1_mod'
                break

        for i in reversed(range(0, path[-1])):
            if ((not dependency[i]['label'].endswith('CNJ')) and dependency[i]['state_label'] != 'entity' and
                        dependency[i]['head'] == path[-1]):
                arg2_mod = self._get_morp_items(i, nlp_result)
                dependency[i]['state_label'] = 'arg2_mod'
                break

        context_lemma = []
        left_limit = min(path[0], path[-1]) - 2 if (min(path[0], path[-1])) > 3 else 0
        right_limit = max(path[0], path[-1]) + 2 + 1
        if (right_limit >= len(dependency)):
            right_limit = len(dependency)
        for i in range(left_limit, right_limit):
            if (dependency[i]['state_label'] == ''):
                context_lemma.extend(self._get_morp_items(i, nlp_result))

        dependency_path_morp = []
        words = nlp_result['word']
        sbj_label = dependency[path[0]]['label']
        obj_label = dependency[path[-1]]['label']
        sbj_josa = ''
        obj_josa = ''
        for morp_num in range(nlp_result['word'][path[0]]['begin'], nlp_result['word'][path[0]]['end'] + 1):
            if morp[morp_num]['type'].startswith('J'):
                sbj_josa = morp[morp_num]['lemma']

        for morp_num in range(nlp_result['word'][path[-1]]['begin'], nlp_result['word'][path[-1]]['end'] + 1):
            if morp[morp_num]['type'].startswith('J'):
                obj_josa = morp[morp_num]['lemma']

        for i in range(len(path) - 1):
            direction = 'down' if graph[path[i]][path[i + 1]] == 2 else 'up'
            dependency_path.append(direction + '||')
            if (i < len(path) - 2):
                dependency_path[-1] = direction + '||' + dependency[path[i + 1]]['label']
                dependency_path.append(words[path[i + 1]]['text'])
                dependency_path_morp.extend(self._get_morp_items(path[i + 1], nlp_result))

        morp = nlp_result['morp']
        sbj_NE = obj_NE = 'NONE'
        for word in nlp_result['NE']:
            word_st = morp[word['begin']]['position']
            word_ed = morp[word['end'] + 1]['position'] - 1 if word['end'] + 1 < len(morp) else 10000
            if sbj_byte_loc >= word_st and sbj_byte_loc <= word_ed:
                sbj_NE = word['type']
            if obj_byte_loc >= word_st and obj_byte_loc <= word_ed:
                obj_NE = word['type']

        result = {
            'sent': sent,
            'origin_sent': origin_sent,
            'template_sent': new_template_sent,
            'sbj': sbj,
            'obj': obj,
            'sbj_ne': sbj_NE,
            'obj_ne': obj_NE,
            'sbj_label': sbj_label,
            'obj_label': obj_label,
            'sbj_josa': sbj_josa,
            'obj_josa': obj_josa,
            'dependency': dependency_path,
            'dependency_morp': dependency_path_morp,
            'arg1_mod': arg1_mod,
            'arg2_mod': arg2_mod,
            'context_lemma': context_lemma
        }
        return result

if __name__ == '__main__':
    pass