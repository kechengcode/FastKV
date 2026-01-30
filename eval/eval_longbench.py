import os
import json
import argparse
import numpy as np

from eval.metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--longbench_e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()
    
    dataset_list = list(dataset2metric.keys())
    
    results_list = [
        ["dataset"],
        ["fullkv"],
        ["streamingllm"],
        ["h2o"],
        ["snapkv"],
        ["pyramidinfer"],
        ["gemfilter"],
        ["fastkv"]
    ]
    
    for dataset in dataset_list:
        results_list[0].append(dataset)
        
        for idx, method in enumerate(["fullkv", "streamingllm", "h2o", "snapkv", "pyramidinfer", "gemfilter", "fastkv"]):
            try:
                args.method = method
                args.dataset = dataset
                args.eval_file = os.path.join(args.results_dir, dataset, f"{method}.json")
                
                # try:
                
                scores = dict()
                # if args.longbench_e:
                #     path = f"pred_e/{args.model}/"
                # else:
                #     path = f"pred_e/{args.model}/"
                # all_files = os.listdir(path)
                # print("Evaluating on:", all_files)
                
                # for filename in all_files:
                    # if not filename.endswith("jsonl"):
                    #     continue
                predictions, answers, lengths = [], [], []
                # dataset = filename.split('.')[0]
                with open(args.eval_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            predictions.append(data["pred"])
                            answers.append(data["answers"])
                            all_classes = data["all_classes"]
                            if "length" in data:
                                lengths.append(data["length"])
                        except:
                            print("error")
                if args.longbench_e:
                    score = scorer_e(args.dataset, predictions, answers, lengths, all_classes)
                else:
                    score = scorer(args.dataset, predictions, answers, all_classes)
                    if args.dataset == 'qasper':
                        score_e = scorer_e(args.dataset, predictions, answers, lengths, all_classes)
                scores[args.dataset] = score
                    # if dataset == 'qasper':
                    #     scores[dataset + '_e'] = score_e
                    
                # if args.longbench_e:
                #     out_path = f"H2O/results/{args.model}/result.json"
                # else:
                #     out_path = f"H2O/results/{args.model}/result.json"
                    # out_path_e = f"pred/{args.model}/result_e.json"
                    # with open(out_path_e, "w") as f:
                    #     json.dump(score_e, f, ensure_ascii=False, indent=4)
                    
                output_dir = os.path.dirname(args.eval_file)
                
                results_list[idx+1].append(score)
                
                with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                    json.dump(scores, f, ensure_ascii=False, indent=4)
            
                print(f"dataset {args.dataset} method {args.method} scores {scores}")
            except:

                results_list[idx+1].append(-1)
                # print(f"dataset {args.dataset} method {args.method} scores {None}")
                
    import csv
    with open(os.path.join(args.results_dir, f"results.csv"), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(results_list)

    # Print Excel-friendly CSV summary at the end for easy paste
    print("\nExcel-friendly summary (CSV):")
    header = ["method"] + dataset_list
    print(",".join(header))
    for idx, method in enumerate(["fullkv", "streamingllm", "h2o", "snapkv", "pyramidinfer", "gemfilter", "fastkv"]):
        # results_list[idx+1] is the row for this method; skip the first cell (method name)
        values = []
        for val in results_list[idx+1][1:]:
            if isinstance(val, (int, float, np.floating)):
                values.append(f"{float(val):.2f}")
            else:
                try:
                    values.append(f"{float(val):.2f}")
                except:
                    values.append(str(val))
        print(",".join([method] + values))

    # ==========================================
    # Added: Aggregated Category Summary Table
    # ==========================================
    print("\n" + "="*50)
    print("Aggregated Category Summary (Table 2 Format)")
    print("="*50)

    # Define Categories (Standard LongBench-E / English subset)
    categories = {
        "Single-Doc QA": ["narrativeqa", "qasper", "multifieldqa_en"],
        "Multi-Doc QA": ["hotpotqa", "2wikimqa", "musique"],
        "Summarization": ["gov_report", "qmsum", "multi_news"],
        "Few-shot": ["trec", "triviaqa", "samsum"],
        "Synthetic": ["passage_retrieval_en", "passage_count"],
        "Code": ["lcc", "repobench-p"]
    }

    # Prepare Header
    cat_headers = list(categories.keys()) + ["Avg"]
    print(f"{'Method':<15} | " + " | ".join([f"{h:<13}" for h in cat_headers]))
    print("-" * 120)

    # Helper to find index of dataset in dataset_list
    def get_score(method_row, dataset_name):
        try:
            idx = dataset_list.index(dataset_name)
            # method_row has "method_name" at index 0, so score is at idx+1
            val = method_row[idx+1]
            if isinstance(val, (int, float, np.floating)):
                return float(val)
            return float(val) # Try cast
        except:
            return None

    for idx, method in enumerate(["fullkv", "streamingllm", "h2o", "snapkv", "pyramidinfer", "gemfilter", "fastkv"]):
        row_data = results_list[idx+1]
        
        cat_scores = []
        valid_overall_scores = []
        
        for cat, datasets in categories.items():
            scores = []
            for ds in datasets:
                s = get_score(row_data, ds)
                if s is not None and s >= 0: # Check for valid score
                    scores.append(s)
                    valid_overall_scores.append(s)
            
            if scores:
                avg_score = sum(scores) / len(scores)
                cat_scores.append(f"{avg_score:.2f}")
            else:
                cat_scores.append("-")
        
        # Calculate Total Avg (Macro average over valid tasks in these categories)
        if valid_overall_scores:
            total_avg = sum(valid_overall_scores) / len(valid_overall_scores)
            cat_scores.append(f"{total_avg:.2f}")
        else:
            cat_scores.append("-")

        # Print Row
        print(f"{method:<15} | " + " | ".join([f"{s:<13}" for s in cat_scores]))
