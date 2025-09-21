import json
import os

def extract_top100k_content(input_file, output_file, max_count=100000):
    """
    从JSON Lines文件中提取前100000条有效content，保存到纯文本文件
    :param input_file: 输入JSON Lines文件路径
    :param output_file: 输出提取结果的路径
    :param max_count: 最大提取数量（默认100000）
    """
    if not os.path.exists(input_file):
        print(f"错误：输入文件不存在 → {input_file}")
        return

    extracted_count = 0  # 已成功提取的有效content数量
    total_line = 0       # 已读取的总行数
    error_count = 0      # 解析失败的行数

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        print(f"开始提取，目标：前{max_count}条有效content...")
        
        for line in f_in:
            total_line += 1
            line = line.strip()
            if not line:
                continue  # 跳过空行

            try:
                # 解析当前行的JSON对象
                json_obj = json.loads(line)
                # 提取非空的content（空content不计数）
                content = json_obj.get("content", "").strip()
                if not content:
                    continue  # 跳过无content或content为空的条目

                # 写入输出文件（每个content后加两个换行区分）
                f_out.write(content + "\n\n")
                extracted_count += 1

                # 进度提示（每提取10000条打印一次，避免刷屏）
                if extracted_count % 10000 == 0:
                    print(f"已提取 {extracted_count}/{max_count} 条content")

                # 提取满100000条，立即停止
                if extracted_count >= max_count:
                    print(f"\n已达到目标数量（{max_count}条），停止提取")
                    break

            except json.JSONDecodeError as e:
                print(f"警告：第{total_line}行JSON解析失败 → {str(e)[:50]}...")
                error_count += 1
            except Exception as e:
                print(f"警告：第{total_line}行处理出错 → {str(e)[:50]}...")
                error_count += 1

    # 最终统计结果
    print(f"\n===== 提取完成 =====")
    print(f"总读取行数：{total_line}")
    print(f"成功提取content：{extracted_count} 条（目标：{max_count}条）")
    print(f"解析失败行数：{error_count}")
    print(f"结果保存至：{output_file}")
    # 提示是否文件不足100000条
    if extracted_count < max_count:
        print(f"⚠️  注意：文件中有效content不足{max_count}条，已提取全部可用条目")

if __name__ == "__main__":
    # --------------------------
    # 请修改为你的实际路径
    # --------------------------
    INPUT_FILE = "news_original.jsonl"    # 你的原始文件（每行一个JSON对象）
    OUTPUT_FILE = "top20k_content.txt"  # 提取结果保存路径

    # 执行提取（默认提取前20000条，如需调整数量，修改max_count参数即可）
    extract_top100k_content(INPUT_FILE, OUTPUT_FILE, max_count=20000)