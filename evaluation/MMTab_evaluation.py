import json
import re
import string
import argparse
from collections import defaultdict
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import tqdm
from project_config.config import cfg
from sacrebleu.metrics import BLEU
from utils.helpers import (
    _normalize,
    _clean_num,
    extract_tqa_answer_list,
    _get_question_text,
    _get_sorted_groups,
    _extract_first_sentence_for_fetaqa,
    _extract_fetaqa_answer_for_evaluation,
    _normalize_answer_robustly,
    _fallback_containment_check,
)


def read_and_group_predictions(file_path, prediction_field, eval_mode):
    preds = []
    with open(file_path, encoding="utf-8") as f:
        for line in tqdm.tqdm(f, desc="Reading predictions"):
            it = json.loads(line)
            it["output"] = it.get(prediction_field, "")
            preds.append(it)
    print("Predicted Sample Number:", len(preds))
    bm_map = defaultdict(lambda: defaultdict(list))
    for it in preds:
        dataset, task = it["category"].split("_for_")
        benchmark = (
            dataset
            if task
            not in [
                "TSD",
                "TCL",
                "RCE",
                "MCD",
                "TCE",
                "TR",
                "OOD_TSD",
                "OOD_TCL",
                "OOD_RCE",
                "OOD_TCE",
            ]
            else task
        )
        group = (
            it.get("selected_expert") or "overall"
            if eval_mode == "per_expert"
            else "overall"
        )
        bm_map[benchmark][group].append(it)
    print("\n--- Data Grouping Summary ---")
    for bm, groups in bm_map.items():
        print(f"Benchmark: {bm}")
        if eval_mode == "per_expert":
            total_samples = sum((len(lst) for lst in groups.values()))
            for g, lst in _get_sorted_groups(groups):
                percentage = len(lst) / total_samples * 100 if total_samples > 0 else 0
                print(f"  - Group: {g}, test data num: {len(lst)} ({percentage:.1f}%)")
        else:
            for g, lst in _get_sorted_groups(groups):
                print(f"  - Group: {g}, test data num: {len(lst)}")
    print("-----------------------------\n")
    return bm_map


def evaluate_text_generation_questions(pred_list, id2item, return_metrics=False):
    bleu = BLEU()
    outs, refs = ([], [])
    for it in pred_list:
        qid = it["question_id"]
        gold = id2item[qid].get("output", id2item[qid].get("target_answer", ""))
        raw_pred = it["output"]
        selected_expert = it.get("selected_expert", "TextExpert")
        pred = _extract_fetaqa_answer_for_evaluation(raw_pred, selected_expert)
        outs.append(str(pred))
        refs.append(str(gold))
    bleu_score = bleu.corpus_score(outs, [refs]).score
    if return_metrics:
        return {"bleu": bleu_score, "total": len(pred_list)}
    print(f"  BLEU: {bleu.corpus_score(outs, [refs]).format(width=2)}")


def _tqa_core_eval(
    pred_item_list, id2item, benchmark_name, is_hitab=False, return_metrics=False
):
    correct, wrong, failed = ([], [], [])
    for it in pred_item_list:
        try:
            qid = it["question_id"]
            raw_gold_list = id2item[qid]["answer_list"]
            model_output = it["output"]
            pred_list = extract_tqa_answer_list(model_output)
            processed_gold_list = []
            for gold in raw_gold_list:
                processed_gold_list.extend(str(gold).split("|"))
            pred_set = {_normalize_answer_robustly(p) for p in pred_list}
            gold_set = {_normalize_answer_robustly(g) for g in processed_gold_list}
            is_correct = gold_set == pred_set
            if not is_correct:
                is_correct = _fallback_containment_check(
                    model_output, processed_gold_list
                )
            if is_correct:
                correct.append(it)
            else:
                wrong.append(it)
        except Exception as e:
            failed.append(it)
    total = len(pred_item_list)
    acc = len(correct) / total if total else 0
    if return_metrics:
        return {
            "accuracy": acc,
            "total": total,
            "correct": len(correct),
            "failed": len(failed),
        }
    print(f"\nBenchmark : {benchmark_name}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Total     : {total}")
    if failed:
        print(f"Failed    : {len(failed)} (parsing errors)")
    print("-" * 20)


def evaluate_single_benchmark(bm, bm_groups, id2item, eval_mode):
    if bm not in bm_groups:
        return None
    results = {}
    total_samples_in_benchmark = (
        sum((len(plist) for plist in bm_groups[bm].values()))
        if eval_mode == "per_expert"
        else 0
    )
    if eval_mode == "per_expert":
        all_preds = []
        for grp, plist in _get_sorted_groups(bm_groups[bm]):
            all_preds.extend(plist)
        if all_preds:
            if bm == "FeTaQA":
                metrics = evaluate_text_generation_questions(
                    all_preds, id2item, return_metrics=True
                )
                results["Overall"] = {
                    "metric": metrics["bleu"],
                    "total": metrics["total"],
                    "percentage": 100.0,
                }
            elif bm in ("TABMWP", "WTQ", "TAT-QA", "TabFact", "InfoTabs"):
                metrics = _tqa_core_eval(
                    all_preds, id2item, bm, is_hitab=False, return_metrics=True
                )
                results["Overall"] = {
                    "metric": metrics["accuracy"],
                    "total": metrics["total"],
                    "percentage": 100.0,
                }
            elif bm == "HiTab":
                metrics = _tqa_core_eval(
                    all_preds, id2item, bm, is_hitab=True, return_metrics=True
                )
                results["Overall"] = {
                    "metric": metrics["accuracy"],
                    "total": metrics["total"],
                    "percentage": 100.0,
                }
    for grp, plist in _get_sorted_groups(bm_groups[bm]):
        if not plist:
            continue
        percentage = (
            len(plist) / total_samples_in_benchmark * 100
            if total_samples_in_benchmark > 0
            else 100.0
        )
        if bm == "FeTaQA":
            metrics = evaluate_text_generation_questions(
                plist, id2item, return_metrics=True
            )
            results[grp] = {
                "metric": metrics["bleu"],
                "total": metrics["total"],
                "percentage": percentage,
            }
        elif bm in ("TABMWP", "WTQ", "TAT-QA", "TabFact", "InfoTabs"):
            metrics = _tqa_core_eval(
                plist, id2item, bm, is_hitab=False, return_metrics=True
            )
            results[grp] = {
                "metric": metrics["accuracy"],
                "total": metrics["total"],
                "percentage": percentage,
            }
        elif bm == "HiTab":
            metrics = _tqa_core_eval(
                plist, id2item, bm, is_hitab=True, return_metrics=True
            )
            results[grp] = {
                "metric": metrics["accuracy"],
                "total": metrics["total"],
                "percentage": percentage,
            }
    return results


def generate_markdown_table(all_results, eval_mode):
    benchmarks = ["TABMWP", "WTQ", "HiTab", "TAT-QA", "TabFact", "InfoTabs", "FeTaQA"]
    experts = ["Fusion", "VLMExpert", "TextExpert"]
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 80)
    if eval_mode == "per_expert":
        print("\n### Overall Performance (All Experts Combined)")
        print("| Dataset | Metric | Total Samples |")
        print("|---------|--------|---------------|")
        for bm in benchmarks:
            if bm in all_results and "Overall" in all_results[bm]:
                result = all_results[bm]["Overall"]
                metric_name = "BLEU-4" if bm == "FeTaQA" else "Accuracy"
                metric_value = f"{result['metric']:.4f}"
                print(f"| {bm} | {metric_value} ({metric_name}) | {result['total']} |")
        print(f"\n### Expert-wise Performance")
        header = "| Dataset |"
        for expert in experts:
            header += f" {expert} (Acc/%) |"
        print(header)
        separator = "|---------|"
        for _ in experts:
            separator += "----------|"
        print(separator)
        for bm in benchmarks:
            if bm not in all_results:
                continue
            row = f"| {bm} |"
            for expert in experts:
                if expert in all_results[bm]:
                    result = all_results[bm][expert]
                    if bm == "FeTaQA":
                        metric_str = (
                            f"{result['metric']:.3f}({result['percentage']:.1f}%)"
                        )
                    else:
                        metric_str = (
                            f"{result['metric']:.3f}({result['percentage']:.1f}%)"
                        )
                else:
                    metric_str = "N/A"
                row += f" {metric_str} |"
            print(row)
        print(f"\n### Summary Statistics")
        print("| Expert | Avg Accuracy | Total Samples | Datasets |")
        print("|--------|--------------|---------------|----------|")
        for expert in experts:
            accuracies = []
            total_samples = 0
            datasets = []
            for bm in benchmarks:
                if bm in all_results and expert in all_results[bm]:
                    if bm != "FeTaQA":
                        accuracies.append(all_results[bm][expert]["metric"])
                    total_samples += all_results[bm][expert]["total"]
                    datasets.append(bm)
            avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0
            datasets_str = ", ".join(datasets) if datasets else "N/A"
            print(f"| {expert} | {avg_acc:.4f} | {total_samples} | {datasets_str} |")
    else:
        print("\n### Overall Results")
        print("| Dataset | Metric | Total Samples |")
        print("|---------|--------|---------------|")
        for bm in benchmarks:
            if bm in all_results and "overall" in all_results[bm]:
                result = all_results[bm]["overall"]
                metric_name = "BLEU-4" if bm == "FeTaQA" else "Accuracy"
                metric_value = f"{result['metric']:.4f}"
                print(f"| {bm} | {metric_value} ({metric_name}) | {result['total']} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, default=None)
    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--prediction_field", type=str, default="output")
    parser.add_argument("--benchmark", type=str, default="HiTab")
    parser.add_argument(
        "--eval_mode", choices=["overall", "per_expert"], default="overall"
    )
    parser.add_argument(
        "--all_benchmarks",
        action="store_true",
        help="Evaluate all 7 benchmarks and generate markdown summary table",
    )
    args = parser.parse_args()
    if args.pred_file is None:
        out_dir = str(cfg["DATA"].get("INFERENCE_OUTPUT_DIR", os.path.join("output", "main")))
        args.pred_file = os.path.join(out_dir, "inference_results.jsonl")
    if args.eval_data is None:
        args.eval_data = os.path.join("evaluation", "eval_data.json")
    bm_groups = read_and_group_predictions(
        args.pred_file, args.prediction_field, args.eval_mode
    )
    with open(args.eval_data, encoding="utf-8") as f:
        gt = json.load(f)
    id2item = {it.get("item_id", it.get("question_id")): it for it in gt}
    print(f"\nMMTab-eval ground truth items: {len(id2item)}")
    if args.all_benchmarks:
        benchmarks = [
            "TABMWP",
            "WTQ",
            "HiTab",
            "TAT-QA",
            "TabFact",
            "InfoTabs",
            "FeTaQA",
        ]
        all_results = {}
        for bm in benchmarks:
            if bm in bm_groups:
                print(f"\n===== Evaluating Benchmark: {bm} =====")
                results = evaluate_single_benchmark(
                    bm, bm_groups, id2item, args.eval_mode
                )
                if results:
                    all_results[bm] = results
                    total_samples_in_benchmark = (
                        sum((len(plist) for plist in bm_groups[bm].values()))
                        if args.eval_mode == "per_expert"
                        else 0
                    )
                    if args.eval_mode == "per_expert" and "Overall" in results:
                        print(
                            f"\n--- Overall (All Experts Combined, Samples: {results['Overall']['total']}) ---"
                        )
                        if bm == "FeTaQA":
                            evaluate_text_generation_questions(
                                [
                                    it
                                    for grp, plist in _get_sorted_groups(bm_groups[bm])
                                    for it in plist
                                ],
                                id2item,
                            )
                        elif bm in ("TABMWP", "WTQ", "TAT-QA", "TabFact", "InfoTabs"):
                            _tqa_core_eval(
                                [
                                    it
                                    for grp, plist in _get_sorted_groups(bm_groups[bm])
                                    for it in plist
                                ],
                                id2item,
                                bm,
                                is_hitab=False,
                            )
                        elif bm == "HiTab":
                            _tqa_core_eval(
                                [
                                    it
                                    for grp, plist in _get_sorted_groups(bm_groups[bm])
                                    for it in plist
                                ],
                                id2item,
                                bm,
                                is_hitab=True,
                            )
                    for grp, plist in _get_sorted_groups(bm_groups[bm]):
                        if not plist:
                            continue
                        if (
                            args.eval_mode == "per_expert"
                            and total_samples_in_benchmark > 0
                        ):
                            percentage = len(plist) / total_samples_in_benchmark * 100
                            print(
                                f"\n--- Group: {grp} (Samples: {len(plist)}, {percentage:.1f}%) ---"
                            )
                        else:
                            print(f"\n--- Group: {grp} (Samples: {len(plist)}) ---")
                        if bm == "FeTaQA":
                            evaluate_text_generation_questions(plist, id2item)
                        elif bm in ("TABMWP", "WTQ", "TAT-QA", "TabFact", "InfoTabs"):
                            _tqa_core_eval(plist, id2item, bm, is_hitab=False)
                        elif bm == "HiTab":
                            _tqa_core_eval(plist, id2item, bm, is_hitab=True)
            else:
                print(f"\nNo prediction items for benchmark '{bm}'.")
        generate_markdown_table(all_results, args.eval_mode)
    else:
        bm = args.benchmark
        if bm not in bm_groups:
            print(f"\nNo prediction items for benchmark '{bm}'.")
            return
        print(f"\n===== Evaluating Benchmark: {bm} =====")
        total_samples_in_benchmark = (
            sum((len(plist) for plist in bm_groups[bm].values()))
            if args.eval_mode == "per_expert"
            else 0
        )
        if args.eval_mode == "per_expert":
            all_preds = []
            for grp, plist in _get_sorted_groups(bm_groups[bm]):
                all_preds.extend(plist)
            if all_preds:
                print(
                    f"\n--- Overall (All Experts Combined, Samples: {len(all_preds)}) ---"
                )
                if bm == "FeTaQA":
                    evaluate_text_generation_questions(all_preds, id2item)
                elif bm in ("TABMWP", "WTQ", "TAT-QA", "TabFact", "InfoTabs"):
                    _tqa_core_eval(all_preds, id2item, bm, is_hitab=False)
                elif bm == "HiTab":
                    _tqa_core_eval(all_preds, id2item, bm, is_hitab=True)
                else:
                    print(f"  No evaluation logic implemented for benchmark '{bm}'.")
        for grp, plist in _get_sorted_groups(bm_groups[bm]):
            if not plist:
                continue
            if args.eval_mode == "per_expert" and total_samples_in_benchmark > 0:
                percentage = len(plist) / total_samples_in_benchmark * 100
                print(
                    f"\n--- Group: {grp} (Samples: {len(plist)}, {percentage:.1f}%) ---"
                )
            else:
                print(f"\n--- Group: {grp} (Samples: {len(plist)}) ---")
            if bm == "FeTaQA":
                evaluate_text_generation_questions(plist, id2item)
            elif bm in ("TABMWP", "WTQ", "TAT-QA", "TabFact", "InfoTabs"):
                _tqa_core_eval(plist, id2item, bm, is_hitab=False)
            elif bm == "HiTab":
                _tqa_core_eval(plist, id2item, bm, is_hitab=True)
            else:
                print(f"  No evaluation logic implemented for benchmark '{bm}'.")


if __name__ == "__main__":
    main()
