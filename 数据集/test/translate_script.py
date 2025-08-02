# import json
# from translate import Translator
#
# # 假设你的文件名为 data.json
# file_path = '../sample_text.json'
#
# # 读取文件
# with open(file_path, 'r', encoding='utf-8') as file:
#     data = json.load(file)
#
# # 初始化翻译器
# translator = Translator(to_lang="zh")
#
# # 翻译所有 text 内容
# for key, value in data.items():
#     if 'text' in value:
#         try:
#             # 翻译 text 内容
#             translated_text = translator.translate(value['text'])
#             # 更新数据
#             value['text'] = translated_text
#             print(f"翻译结果：{translated_text}")
#         except Exception as e:
#             print(f"翻译出错: {e}")
#             value['text'] = f"翻译失败: {value['text']}"
#
# # 将翻译后的内容保存到新文件
# output_file_path = './sample_text.json'
# with open(output_file_path, 'w', encoding='utf-8') as file:
#     json.dump(data, file, ensure_ascii=False, indent=4)
#
# print(f"翻译完成，结果已保存到 {output_file_path}")

# import json
# from translate import Translator
#
# # 初始化翻译器
# translator = Translator(to_lang="zh")
#
# # 假设你的文件名为 data.json
# input_file_path = '../sample_entity.json'
# output_file_path = './sample_entity.json'
#
# # 读取文件
# with open(input_file_path, 'r', encoding='utf-8') as file:
#     data = json.load(file)
#
# # 遍历 JSON 数据，翻译所有 name 字段
# for key, items in data.items():
#     for item in items:
#         if 'name' in item:
#             try:
#                 # 翻译 name 字段
#                 translated_name = translator.translate(item['name'])
#                 item['name'] = translated_name  # 更新翻译后的 name 字段
#                 print(translated_name)
#             except Exception as e:
#                 print(f"翻译出错: {e}")
#                 print(f"原文: {item['name']}")
#                 print(f"翻译失败")
#                 print("-" * 40)
#
# # 将翻译后的内容保存到新文件
# with open(output_file_path, 'w', encoding='utf-8') as file:
#     json.dump(data, file, ensure_ascii=False, indent=4)
#
# print(f"翻译完成，结果已保存到 {output_file_path}")



import json

# 假设你的第一个文件名为 data1.json
file_path1 = './correct.json'
# 假设你的第二个文件名为 data2.json
file_path2 = './test_text.json'
# 合并后的文件名为 merged_data.json
output_file_path = './translated_data.json'

# 读取第一个文件
with open(file_path1, 'r', encoding='utf-8') as file1:
    data1 = json.load(file1)

# 读取第二个文件
with open(file_path2, 'r', encoding='utf-8') as file2:
    data2 = json.load(file2)

ans = 0
for key,item in data1.items():
    ans += len(item)

print(ans)


for key, item in data2.items():
    if key not in data1:
        print(key)

# # 合并两个 JSON 数据
# merged_data = {}
#
# # 遍历第一个文件的内容
# for key, items in data1.items():
#     merged_data[key] = items  # 将第一个文件的内容添加到合并后的字典中
#
# # 遍历第二个文件的内容
# for key, item in data2.items():
#     if key in merged_data:
#         # 如果键已经存在于合并后的字典中，将 text 字段添加到对应的列表中
#         # merged_data[key].append(item['text'])
#         merged_data[key]['text_en'] = item['text']
#         merged_data[key]['entity'] = ''
#     else:
#         # 如果键不存在于合并后的字典中，直接添加
#         merged_data[key] = [item]
#
# # 将合并后的数据保存到新文件
# with open(output_file_path, 'w', encoding='utf-8') as file:
#     json.dump(merged_data, file, ensure_ascii=False, indent=4)
#
# print(f"合并完成，结果已保存到 {output_file_path}")