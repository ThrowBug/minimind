import argparse
import random
import warnings
import numpy as np
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *

warnings.filterwarnings('ignore')


def calculate_ppl_for_lengths(text, lengths, model, tokenizer, device):
    print("计算困惑度...")
    print("=" * 60)
    for length in lengths:
        truncated_text = text[:length]
        inputs = tokenizer(truncated_text, return_tensors='pt', truncation=True).to(device)
        input_ids = inputs['input_ids']
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        # 将logits和labels对齐：logits[:-1] 对应 labels[1:]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        # 计算交叉熵损失
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # 计算困惑度
        ppl = math.exp(loss.item())
        num_tokens = input_ids.shape[1]
        avg_loss = loss.item()
        print(f"Length: {length:5d} chars, Tokens: {num_tokens:4d}, Avg Loss: {avg_loss:.4f}, Perplexity: {ppl:.4f}")
    print("=" * 60)


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    print(f'MiniMind模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    return model.eval().to(args.device), tokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate MiniMind Long Text Perplexity")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--long_text_file', default='./dataset/xiyouji.txt', type=str, help='长文本文件路径')
    args = parser.parse_args()
    
    model, tokenizer = init_model(args)
    try:
        with open(args.long_text_file, 'r', encoding='utf-8') as file:
            text = file.read()
        print(f"Text length: {len(text)} characters")
    except FileNotFoundError:
        print(f"错误：找不到文件 {args.long_text_file}")
        return

    start_length, max_length = 500, 10000
    lengths = [i for i in range(start_length, max_length + 1, 1000)]
    # 计算不同长度的困惑度
    calculate_ppl_for_lengths(text, lengths, model, tokenizer, args.device)


if __name__ == "__main__":
    main()
