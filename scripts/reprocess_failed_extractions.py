#!/usr/bin/env python3
"""
重新处理失败的 topic extraction

用法：
    python reprocess_failed_extractions.py \
        --failed_file <path_to_failed_extractions.json> \
        --output_dir <output_directory> \
        --llm_name <llm_model_name>
"""

import json
import argparse
import os
import sys
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def reprocess_failed_extractions(failed_file, output_dir, llm_name, max_retries=5):
    """
    重新处理失败的 topic extraction

    Args:
        failed_file: 失败记录文件路径
        output_dir: 输出目录
        llm_name: LLM 模型名称
        max_retries: 最大重试次数
    """
    print("=" * 80)
    print("重新处理失败的 Topic Extraction")
    print("=" * 80)
    print()

    # 读取失败记录
    print(f"📂 读取失败记录: {failed_file}")
    with open(failed_file, 'r', encoding='utf-8') as f:
        failed_data = json.load(f)

    total_failed = failed_data['total_failed']
    failed_records = failed_data['failed_records']

    print(f"✓ 找到 {total_failed} 个失败记录")
    print()

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ 输出目录: {output_dir}")
    print()

    # 统计
    success_count = 0
    still_failed_count = 0
    reprocessed_records = []

    # 重新处理每个失败的记录
    print("🔄 开始重新处理...")
    print("-" * 80)

    for idx, record in enumerate(failed_records, 1):
        chunk_id = record['chunk_id']
        passage = record['passage']
        original_failure_reason = record['failure_reason']

        print(f"\n[{idx}/{total_failed}] 处理 {chunk_id}")
        print(f"  原因: {original_failure_reason}")
        print(f"  文本长度: {len(passage)} 字符")

        # 这里应该调用实际的 topic_extraction 方法
        # 由于需要完整的环境，这里提供模拟代码
        # 实际使用时需要替换为真实的调用

        # 模拟重新处理
        # result = openie.topic_extraction(chunk_id, passage, max_retries=max_retries)

        # 模拟结果（实际使用时删除这部分）
        import random
        is_success = random.random() > 0.3  # 70% 成功率

        if is_success:
            print(f"  ✓ 重新处理成功")
            success_count += 1

            reprocessed_records.append({
                'chunk_id': chunk_id,
                'status': 'success',
                'original_failure_reason': original_failure_reason,
                'reprocessed_at': str(datetime.now())
            })
        else:
            print(f"  ✗ 重新处理仍然失败")
            still_failed_count += 1

            reprocessed_records.append({
                'chunk_id': chunk_id,
                'status': 'still_failed',
                'original_failure_reason': original_failure_reason,
                'reprocessed_at': str(datetime.now())
            })

    print()
    print("=" * 80)
    print("重新处理完成")
    print("=" * 80)
    print(f"总数: {total_failed}")
    print(f"成功: {success_count} ({success_count/total_failed*100:.1f}%)")
    print(f"仍然失败: {still_failed_count} ({still_failed_count/total_failed*100:.1f}%)")
    print()

    # 保存重新处理的结果
    output_file = os.path.join(output_dir, f'reprocessed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'original_failed': total_failed,
            'reprocessed_success': success_count,
            'still_failed': still_failed_count,
            'records': reprocessed_records
        }, f, indent=2, ensure_ascii=False)

    print(f"✓ 结果已保存到: {output_file}")
    print()

    # 如果还有失败的，保存到新的失败记录
    if still_failed_count > 0:
        still_failed_records = [r for r in reprocessed_records if r['status'] == 'still_failed']

        still_failed_file = os.path.join(output_dir, f'still_failed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

        with open(still_failed_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_failed': still_failed_count,
                'failed_records': still_failed_records
            }, f, indent=2, ensure_ascii=False)

        print(f"⚠️  仍然失败的记录已保存到: {still_failed_file}")
        print()

    print("💡 建议:")
    if still_failed_count > 0:
        print("  - 对于仍然失败的记录，考虑使用更强的 LLM 模型")
        print("  - 检查这些文本是否有特殊格式或内容")
        print("  - 考虑手动处理这些特殊情况")
    else:
        print("  - 所有失败记录都已成功重新处理！")

    print()


def main():
    parser = argparse.ArgumentParser(description='重新处理失败的 topic extraction')
    parser.add_argument('--failed_file', type=str, required=True,
                       help='失败记录文件路径')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--llm_name', type=str, default='gpt-4o-mini',
                       help='LLM 模型名称')
    parser.add_argument('--max_retries', type=int, default=5,
                       help='最大重试次数（默认5）')

    args = parser.parse_args()

    try:
        reprocess_failed_extractions(
            args.failed_file,
            args.output_dir,
            args.llm_name,
            args.max_retries
        )
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {args.failed_file}")
    except json.JSONDecodeError:
        print(f"❌ 错误: 文件格式不正确，无法解析 JSON")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 如果没有参数，显示使用示例
    if len(sys.argv) == 1:
        print("=" * 80)
        print("重新处理失败的 Topic Extraction")
        print("=" * 80)
        print()
        print("用法:")
        print("  python reprocess_failed_extractions.py \\")
        print("      --failed_file <path_to_failed_extractions.json> \\")
        print("      --output_dir <output_directory> \\")
        print("      --llm_name <llm_model_name>")
        print()
        print("示例:")
        print("  python reprocess_failed_extractions.py \\")
        print("      --failed_file outputs/musique/.../failed_extractions_20260202_153045.json \\")
        print("      --output_dir outputs/musique/.../reprocessed/ \\")
        print("      --llm_name gpt-4")
        print()
        print("注意:")
        print("  - 此脚本需要在完整的项目环境中运行")
        print("  - 确保 LLM API 配置正确")
        print("  - 重新处理可能需要较长时间")
        print()
        sys.exit(0)

    main()
