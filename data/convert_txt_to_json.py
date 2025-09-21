import json
import re
import os

def txt_to_json(txt_path, output_json_path, max_entries=None):

    if not os.path.exists(txt_path):
        print(f"错误：输入文件不存在 → {txt_path}")
        return

    control_chars = re.compile(r'[\x00-\x1F\x7F]')
    def clean_text(text):
        text = control_chars.sub('', text)  # 清理不可见控制字符
        text = text.strip()                 # 去除首尾空白（避免空字符串）
        return text

    print(f"开始读取文本文件：{txt_path}")
    with open(txt_path, 'r', encoding='utf-8') as f:
        # 按两个换行符拆分，获取所有独立content条目
        raw_entries = f.read().split('\n\n')  # 匹配txt中条目的分隔方式
        # 过滤空条目+清理文本
        valid_entries = []
        for idx, entry in enumerate(raw_entries, start=1):
            cleaned_entry = clean_text(entry)
            if cleaned_entry:  # 只保留非空的有效条目
                valid_entries.append({"text": cleaned_entry})  # 适配GPT2训练格式：{"text": 内容}
                
                if len(valid_entries) % 10000 == 0:
                    print(f"已处理 {len(valid_entries)} 条有效条目")
                
                if max_entries and len(valid_entries) >= max_entries:
                    print(f"\n已达到最大处理数量（{max_entries}条），停止读取")
                    break

    if not valid_entries:
        print("错误：未从txt文件中提取到有效content条目，请检查文件格式！")
        return
    print(f"\n文本文件处理完成，共提取 {len(valid_entries)} 条有效训练数据")

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(valid_entries, f, ensure_ascii=False, indent=2)  # indent=2便于查看，不影响训练

    print(f"\n===== 转换完成 =====")
    print(f"GPT2训练文件：{output_json_path}")
    print(f"训练数据条数：{len(valid_entries)}")
    print(f"每条数据格式：{{'text': '清理后的content内容'}}")
    print(f"可直接用于原train.py脚本训练（无需修改代码）")

if __name__ == "__main__":
    TXT_INPUT_PATH = "top20k_content.txt"  # 输入的content文本文件
    JSON_OUTPUT_PATH = "train_20k.json"    # 输出的GPT2训练用JSON文件
    MAX_ENTRIES = None                     # 最大处理条目数

    # 执行转换
    txt_to_json(
        txt_path=TXT_INPUT_PATH,
        output_json_path=JSON_OUTPUT_PATH,
        max_entries=MAX_ENTRIES  
    )