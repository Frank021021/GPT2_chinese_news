import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import trange
from transformers import GPT2LMHeadModel
import datetime


def is_word(word):
    for item in list(word):
        if item not in 'qwertyuiopasdfghjklzxcvbnm':
            return False
    return True


def _is_chinese_char(char):
    cp = ord(char)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  
            (cp >= 0x3400 and cp <= 0x4DBF) or  
            (cp >= 0x20000 and cp <= 0x2A6DF) or  
            (cp >= 0x2A700 and cp <= 0x2B73F) or  
            (cp >= 0x2B740 and cp <= 0x2B81F) or  
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  
        return True
    return False


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, context, length, n_ctx, tokenizer, temperature=1.0, top_k=30, top_p=0.0, repitition_penalty=1.0,
                    device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():
        for _ in trange(length, desc="生成中", leave=False):
            inputs = {'input_ids': generated[0][-(n_ctx - 1):].unsqueeze(0)}
            outputs = model(** inputs)
            next_token_logits = outputs[0][0, -1, :]
            for id in set(generated):
                next_token_logits[id] /= repitition_penalty
            next_token_logits = next_token_logits / temperature
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated.tolist()[0]


def fast_sample_sequence(model, context, length, temperature=1.0, top_k=30, top_p=0.0, device='cpu'):
    inputs = torch.LongTensor(context).view(1, -1).to(device)
    if len(context) > 1:
        _, past = model(inputs[:, :-1], None)[:2]
        prev = inputs[:, -1].view(1, -1)
    else:
        past = None
        prev = inputs
    generate = [] + context
    with torch.no_grad():
        for i in trange(length, desc="生成中", leave=False):
            output = model(prev, past=past)
            output, past = output[:2]
            output = output[-1].squeeze(0) / temperature
            filtered_logits = top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
            generate.append(next_token.item())
            prev = next_token.view(1, 1)
    return generate


def generate(n_ctx, model, context, length, tokenizer, temperature=1, top_k=0, top_p=0.0, repitition_penalty=1.0, device='cpu',
             is_fast_pattern=False):
    if is_fast_pattern:
        return fast_sample_sequence(model, context, length, temperature=temperature, top_k=top_k, top_p=top_p,
                                    device=device)
    else:
        return sample_sequence(model, context, length, n_ctx, tokenizer=tokenizer, temperature=temperature, top_k=top_k, top_p=top_p,
                               repitition_penalty=repitition_penalty, device=device)


def auto_line_break(text, max_line_length=60):
    if not text:
        return ""
    line_break_text = []
    current_length = 0
    for char in text:
        line_break_text.append(char)
        if _is_chinese_char(char) or char.isalnum():
            current_length += 1
        if current_length >= max_line_length:
            if char in ['。', '，', '！', '？', '；', '：', '”', '’', '）', '】', '.', ',', '!', '?', ';', ':']:
                line_break_text.append('\n')
                current_length = 0
            else:
                line_break_text.append('\n')
                current_length = 0
    final_text = ''.join(line_break_text).replace('\n\n', '\n').strip()
    lines = final_text.split('\n')
    cleaned_lines = []
    for line in lines:
        line_stripped = line.strip()
        if line_stripped and line_stripped[0] in ['。', '，', '！', '？', '；', '：']:
            if cleaned_lines:
                cleaned_lines[-1] += line_stripped
            else:
                cleaned_lines.append(line_stripped)
        else:
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    parser.add_argument('--length', default=300, type=int, required=False, help='总生成长度（token数，约中文字数）')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='生成batch size')
    parser.add_argument('--temperature', default=1.0, type=float, required=False, help='生成温度')
    parser.add_argument('--topk', default=8, type=int, required=False, help='top-k过滤')
    parser.add_argument('--topp', default=0.9, type=float, required=False, help='top-p过滤')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='词表路径')
    parser.add_argument('--model_path', default='model/final_model', type=str, required=False, help='模型路径')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    parser.add_argument('--fast_pattern', action='store_true', help='快速生成模式')
    parser.add_argument('--save_samples', action='store_true', help='保存样本到txt')
    parser.add_argument('--save_samples_path', default='generated', type=str, required=False, help='样本保存目录')
    parser.add_argument('--repetition_penalty', default=1.2, type=float, required=False, help='重复惩罚')
    parser.add_argument('--max_accept', default=5, type=int, required=False, help='最多接受的样本数')

    args = parser.parse_args()
    print('参数配置:\n' + args.__repr__())

    if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'使用设备: {device}')

    tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    n_ctx = model.config.n_ctx
    print(f'模型最大上下文长度: {n_ctx} token')

    sample_file = None
    accepted_count = 0
    total_attempts = 0
    max_accept = args.max_accept
    fixed_prefix = None

    if args.save_samples:
        os.makedirs(args.save_samples_path, exist_ok=True)
        sample_file = open(f'{args.save_samples_path}/generated_content.txt', 'a', encoding='utf8')
        print(f'保留的样本将保存至: {args.save_samples_path}/generated_content.txt')

    print(f'\n=== GPT2 文本生成工具 ===')
    print(f'目标：保留{max_accept}个样本，每个样本约{args.length}字，每行约80字\n')
    while True:
        prefix_mode = input('请选择开头模式：1.统一开头 2.不同开头 请输入1或2：').strip()
        if prefix_mode in ['1', '2']:
            break
        print('输入无效，请输入1或2\n')

    if prefix_mode == '1':
        while True:
            fixed_prefix = input('请输入统一开头（至少2个字符）：').strip()
            if len(fixed_prefix) >= 2:
                test_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(fixed_prefix))
                if test_tokens:
                    print(f'已设置统一开头：「{fixed_prefix}」\n')
                    break
                else:
                    print('该开头无法被分词器识别，请更换内容\n')
            else:
                print('开头长度过短，请至少输入2个字符\n')

    while accepted_count < max_accept:
        if prefix_mode == '1':
            raw_prefix = fixed_prefix
            print(f'\n第{total_attempts + 1}次生成（统一开头：「{raw_prefix}」）')
        else:
            raw_prefix = input(f'\n请输入第{total_attempts + 1}个生成开头（回车退出，至少2字符）: ').strip()
            if not raw_prefix:
                print('用户主动退出，程序结束')
                break
            if len(raw_prefix) < 2:
                print('开头长度过短，请至少输入2个字符\n')
                continue
            test_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_prefix))
            if not test_tokens:
                print('该开头无法被分词器识别，请更换内容\n')
                continue

        total_attempts += 1
        prefix_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_prefix))

        print(f'正在生成约{args.length}字内容...')
        full_output = generate(
            n_ctx=n_ctx,
            model=model,
            context=prefix_tokens,
            length=args.length - len(prefix_tokens),
            is_fast_pattern=args.fast_pattern,
            tokenizer=tokenizer,
            temperature=args.temperature,
            top_k=args.topk,
            top_p=args.topp,
            repitition_penalty=args.repetition_penalty,
            device=device
        )

        full_text = tokenizer.convert_ids_to_tokens(full_output)
        for i, item in enumerate(full_text[:-1]):
            if is_word(item) and is_word(full_text[i+1]):
                full_text[i] = item + ' '
        for i, item in enumerate(full_text):
            if item == '[MASK]':
                full_text[i] = ''
            elif item == '[CLS]':
                full_text[i] = '\n\n'
            elif item == '[SEP]':
                full_text[i] = '\n'
            elif item == '[UNK]':
                full_text[i] = '□'
        full_clean = ''.join(full_text).replace('##', '').strip()
        full_with_break = auto_line_break(full_clean, max_line_length=80)
        actual_char_count = len(full_clean.replace(' ', '').replace('\n', ''))

        print(f'\n{"="*50} 生成内容 {total_attempts} {"="*50}')
        print(f'【生成开头】: {raw_prefix}')
        print(f'【实际字符数】: 约{actual_char_count}字')
        print(f'【完整内容】:\n{full_with_break}')
        print(f'{"="*108}\n')

        while True:
            user_choice = input(f'是否保留该样本？(已保留 {accepted_count}/{max_accept}) (y/n/quit): ').strip().lower()
            if user_choice in ['y', 'n', 'quit']:
                break
            print('输入无效，请输入 y/n/quit\n')

        if user_choice == 'quit':
            print('用户主动退出，程序结束')
            break
        if user_choice == 'y':
            accepted_count += 1
            print(f'已保留该样本（{accepted_count}/{max_accept}）\n')
            
            if args.save_samples and sample_file:
                save_content = [
                    f'{"="*50} 保留样本 {accepted_count} {"="*50}',
                    f'生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                    f'生成开头: {raw_prefix}',
                    f'实际字符数: 约{actual_char_count}字',
                    f'完整内容:',
                    full_with_break,
                    f'{"="*108}\n\n'
                ]
                sample_file.write('\n'.join(save_content))
                print(f'样本已保存到文件\n')
        else:
            print(f'已放弃该样本（{accepted_count}/{max_accept}）\n')

    if accepted_count >= max_accept:
        print(f'\n✅ 已保留{max_accept}个样本，程序结束！')

    if args.save_samples and sample_file:
        sample_file.close()
        print(f'📁 样本保存至: {args.save_samples_path}/generated_content_2.txt')


if __name__ == '__main__':
    main()