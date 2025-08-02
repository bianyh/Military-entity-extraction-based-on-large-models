import json

text = json.load(open('./数据集/sample_text.json'))
entities = json.load(open('./数据集/sample_entity.json'))

text_filter = dict()

for i in text:
    if i in entities:
        if entities[i] != []:
            text_filter[i] = text[i]


with open ('./数据集/sample_text_filter.json','w') as f:
    json.dump(text_filter,f)