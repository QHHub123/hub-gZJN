"""
在测试集上评估 BERT NER 模型，并统计非法序列

教学重点：
  1. seqeval 的 entity-level 评估（与 token-level 的区别）
     - token-level：逐个 token 算对错（O 标签太多，accuracy 虚高）
     - entity-level：整个实体span必须完全匹配才算对（更严格，更有意义）
  2. 非法序列统计：量化 CRF vs 线性头的关键差异
     - 非法：I-X 在序列开头，或 B-X 后面跟 I-Y（X≠Y）
     - 线性头通常有几十到几百条非法序列，CRF 始终为 0
  3. 逐类型 F1 分析：哪类实体最难识别

使用方式：
  python evaluate.py                        # 评估 BERT+Linear
  python evaluate.py --use_crf              # 评估 BERT+CRF
  python evaluate.py --split validation     # 在验证集上评估
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from transformers import BertTokenizer
from seqeval.metrics import (
    f1_score, precision_score, recall_score,
    classification_report as seqeval_report,
)

from dataset import build_label_schema, build_dataloaders
from model import build_model

ROOT = Path(__file__).parent.parent.parent
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
DATA_DIR = ROOT / "data" / "peoples_daily"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
LOG_DIR = ROOT / "outputs" / "logs"


def count_illegal_sequences(pred_seqs: list[list[str]]) -> dict:
    """统计非法 BIO 序列数量。

    非法类型：
      - illegal_start：序列以 I-X 开头
      - illegal_transition：B-X 或 I-X 后面跟 I-Y（X≠Y）
    """
    stats = {"illegal_start": 0, "illegal_transition": 0, "total_seqs": len(pred_seqs)}
    for seq in pred_seqs:
        if not seq:
            continue
        # 检查开头
        if seq[0].startswith("I-"):
            stats["illegal_start"] += 1

        # 检查转移
        for i in range(1, len(seq)):
            prev, curr = seq[i - 1], seq[i]
            if curr.startswith("I-"):
                curr_type = curr[2:]
                if prev == "O":
                    stats["illegal_transition"] += 1
                elif prev.startswith("B-") or prev.startswith("I-"):
                    prev_type = prev[2:]
                    if prev_type != curr_type:
                        stats["illegal_transition"] += 1

    total_illegal = stats["illegal_start"] + stats["illegal_transition"]
    stats["total_illegal"] = total_illegal
    return stats


def run_inference(
        model,
        loader,
        id2label: dict,
        device: torch.device,
        use_crf: bool,
) -> tuple[list[list[str]], list[list[str]]]:
    """在 loader 上推理，返回 (all_preds, all_golds)，每个元素为字符串标签列表。"""
    model.eval()
    all_preds = []
    all_golds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            if use_crf:
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
            else:
                logits, _ = model(input_ids, attention_mask, token_type_ids)
                pred_ids_list = logits.argmax(dim=-1).tolist()

            labels_list = labels.cpu().tolist()
            attention_list = attention_mask.cpu().tolist()

            for i in range(len(input_ids)):
                gold_seq = []
                pred_seq = []
                token_labels = labels_list[i]

                for j, gold_id in enumerate(token_labels):
                    if gold_id == -100:
                        continue
                    gold_seq.append(id2label[gold_id])
                    if use_crf:
                        pred_seq.append(id2label.get(pred_ids_list[i][j] if j < len(pred_ids_list[i]) else 0, "O"))
                    else:
                        pred_seq.append(id2label.get(pred_ids_list[i][j], "O"))

                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    return all_preds, all_golds


def predict_custom_texts(
        texts: list[str],
        model,
        tokenizer,
        id2label: dict,
        device: torch.device,
        use_crf: bool,
        max_length: int = 128,
) -> list[dict]:
    """对自定义文本进行序列标注。

    Args:
        texts: 待标注的文本列表
        model: 训练好的模型
        tokenizer: BERT tokenizer
        id2label: ID到标签的映射
        device: 计算设备
        use_crf: 是否使用CRF
        max_length: 最大序列长度

    Returns:
        包含文本、tokens和BIO标签的字典列表
    """
    model.eval()
    results = []

    with torch.no_grad():
        for text in texts:
            encoding = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            token_type_ids = encoding["token_type_ids"].to(device)

            if use_crf:
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
                pred_ids = pred_ids_list[0]
            else:
                logits, _ = model(input_ids, attention_mask, token_type_ids)
                pred_ids = logits.argmax(dim=-1)[0].tolist()

            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            bio_labels = []
            valid_tokens = []

            for j, (token_id, token) in enumerate(zip(input_ids[0], tokens)):
                if token_id == tokenizer.pad_token_id:
                    break
                if token in ["[CLS]", "[SEP]"]:
                    continue

                label_id = pred_ids[j] if j < len(pred_ids) else 0
                bio_label = id2label.get(label_id, "O")
                bio_labels.append(bio_label)
                valid_tokens.append(token)

            results.append({
                "text": text,
                "tokens": valid_tokens,
                "bio_labels": bio_labels,
            })

    return results


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_tag = "crf" if args.use_crf else "linear"
    ckpt_path = CKPT_DIR / f"best_{run_tag}.pt"

    if not ckpt_path.exists():
        print(f"找不到 checkpoint：{ckpt_path}")
        print(f"请先运行：python train.py {'--use_crf' if args.use_crf else ''}")
        return

    # 加载checkpoint文件到指定设备（包含模型权重、训练配置等信息）
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 构建标签体系：获取所有标签列表、标签到ID的映射、ID到标签的映射
    labels, label2id, id2label = build_label_schema()

    # 构建NER模型（选择CRF或Linear头，加载BERT预训练模型，设置分类数量）
    model = build_model(
        use_crf=args.use_crf,
        bert_path=str(args.bert_path),
        num_labels=len(labels),
    ).to(device)

    # 从checkpoint中加载模型的权重参数
    model.load_state_dict(ckpt["state_dict"])
    print(f"加载 checkpoint（epoch={ckpt['epoch']}，val_f1={ckpt['val_entity_f1']:.4f}）")

    # 加载BERT分词器（与预训练模型对应）
    tokenizer = BertTokenizer.from_pretrained(str(args.bert_path))

    # 构建数据加载器（训练集、验证集、测试集）
    _, val_loader, test_loader = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=ckpt["args"].get("max_length", 128),
        data_dir=DATA_DIR,
    )
    loader = val_loader if args.split == "validation" else test_loader
    split_name = args.split

    print(f"\n正在在 [{split_name}] 集上推理...")
    all_preds, all_golds = run_inference(model, loader, id2label, device, args.use_crf)

    # seqeval entity-level 指标
    p = precision_score(all_golds, all_preds)  # 计算精确率（Precision）：预测正确的实体数 / 预测的所有实体数
    r = recall_score(all_golds, all_preds)  # 计算召回率（Recall）：预测正确的实体数 / 真实的所有实体数
    f1 = f1_score(all_golds, all_preds)  # 计算F1分数：精确率和召回率的调和平均数

    print("\n" + "=" * 70)
    print(f"模型：{'BERT + CRF' if args.use_crf else 'BERT + Linear'}  |  评估集：{split_name}")
    print("=" * 70)
    print(f"Entity-level Precision: {p:.4f}")
    print(f"Entity-level Recall:    {r:.4f}")
    print(f"Entity-level F1:        {f1:.4f}")

    print("\n【逐类型 F1】")
    print(seqeval_report(all_golds, all_preds, digits=4))

    # 非法序列统计（核心教学对比点）
    illegal_stats = count_illegal_sequences(all_preds)
    print("【非法 BIO 序列统计】")
    print(f"  总序列数：{illegal_stats['total_seqs']}")
    print(f"  非法开头（I-X 开头）：{illegal_stats['illegal_start']} 条")
    print(f"  非法转移（B-X/I-X → I-Y, X≠Y）：{illegal_stats['illegal_transition']} 条")
    print(f"  合计非法序列：{illegal_stats['total_illegal']} 条")
    pct = illegal_stats["total_illegal"] / max(illegal_stats["total_seqs"], 1) * 100
    if args.use_crf:
        if illegal_stats["total_illegal"] == 0:
            print("  → CRF Viterbi 解码：非法序列 0 条 ✓（转移矩阵已充分学习约束）")
        else:
            print(f"  → CRF 非法序列 {illegal_stats['total_illegal']} 条（{pct:.1f}%）")
            print(f"  → 提示：训练 epoch 不足时转移矩阵尚未收敛；充分训练（3+ epochs）后可降至 0")
    else:
        print(f"  → 线性头约 {pct:.1f}% 的序列含非法转移，充分训练的 CRF 可完全消除")

    # 保存结果 JSON
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "model": "BERT+CRF" if args.use_crf else "BERT+Linear",
        "split": split_name,
        "precision": round(p, 6),
        "recall": round(r, 6),
        "f1": round(f1, 6),
        "illegal_stats": illegal_stats,
    }
    out_path = LOG_DIR / f"eval_{run_tag}_{split_name}.json"
    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存 → {out_path}")

    # ==========================添加自定义文本，进行序列标注==================================== #

    # 自定义文本预测示例
    print("\n" + "=" * 70)
    print("【自定义文本序列标注示例】")
    print("=" * 70)

    custom_texts = [
        "李明在北京大学读书",
        "马云创立了阿里巴巴集团",
        "张华在上海工作",
    ]

    predictions = predict_custom_texts(
        texts=custom_texts,
        model=model,
        tokenizer=tokenizer,
        id2label=id2label,
        device=device,
        use_crf=args.use_crf,
        max_length=ckpt["args"].get("max_length", 128),
    )

    for pred in predictions:
        print(f"\n文本: {pred['text']}")
        print(f"Tokens: {pred['tokens']}")
        print(f"BIO标签: {pred['bio_labels']}")

        # 提取实体
        entities = []
        current_entity = []
        current_type = None

        for token, label in zip(pred['tokens'], pred['bio_labels']):
            if label.startswith('B-'):
                if current_entity:
                    entities.append((current_type, ''.join(current_entity)))
                current_entity = [token]
                current_type = label[2:]
            elif label.startswith('I-') and current_type == label[2:]:
                current_entity.append(token)
            else:
                if current_entity:
                    entities.append((current_type, ''.join(current_entity)))
                    current_entity = []
                    current_type = None

        if current_entity:
            entities.append((current_type, ''.join(current_entity)))

        print(f"识别实体: {entities}")


def parse_args():
    parser = argparse.ArgumentParser(description="评估 BERT NER 模型")
    parser.add_argument("--use_crf", action="store_true")
    parser.add_argument("--bert_path", type=Path, default=BERT_PATH)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--split", choices=["validation", "test"], default="validation",
                        help="CLUE cluener2020 test 集标签未公开，请使用 validation")
    return parser.parse_args()


if __name__ == "__main__":
    main()
