import re


def tokenize_pid(s):
    pattern = r"""
        \(C\)\{\w+\}_\d+         |  
        \(C\)\{\w+\}             |
        \{\w+\}_\d+              |
        \([^)]+\)\{\d+\}         |
        \([^)]+\)                |
        \{[^}]+\}                |
        <_\d+                    |
        _\d+                     |
        \[|\]                    |
        [|&<>(){}]               |
        n                        |
        \w+                      |
        .                        |
    """
    return re.findall(pattern, s, re.VERBOSE)

def is_controller_token(token):
    return bool(
        re.fullmatch(r"\(C\)\{\w+\}_\d+", token) or
        re.fullmatch(r"\(C\)\{\w+\}", token) or
        re.fullmatch(r"\{\w+\}_\d+", token) or
        re.fullmatch(r"_\d+",token)
    )

def clean_tokens(tokens):
    cleaned = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "[":
            nested = []
            depth = 1
            i += 1
            while i < len(tokens):
                if tokens[i] == "[":
                    depth += 1
                elif tokens[i] == "]":
                    depth -= 1
                    if depth == 0:
                        break
                nested.append(tokens[i])
                i += 1
            inner_cleaned = clean_tokens(nested)
            if inner_cleaned:
                cleaned.append("[" + ''.join(inner_cleaned) + "]")
        elif is_controller_token(token) or re.fullmatch(r"<_\d+", token):
            pass  
        else:
            cleaned.append(token)
        i += 1
    return cleaned

def reconstruct_string(tokens):
    return ''.join(tokens)

def normalize(s):
    return s.replace(" ", "").strip()

def compare_input_with_cleaned_prediction(input_str, prediction_str):
    input_tokens = tokenize_pid(input_str)
    prediction_tokens = tokenize_pid(prediction_str)
    cleaned_pred_tokens = clean_tokens(prediction_tokens)
    input_cleaned = reconstruct_string(input_tokens)
    pred_cleaned = reconstruct_string(cleaned_pred_tokens)
    match = normalize(input_cleaned) == normalize(pred_cleaned)
    return match, input_cleaned, pred_cleaned


def extract_controllers(s):
    return set(re.findall(r"\(C\)\{\w+\}_\d+", s))

def check_input_in_prediction(input_str, prediction_str):
    return normalize(input_str) in normalize(prediction_str)


def evaluate_prediction(sample_input, prediction, gt):
    match, cleaned_input, cleaned_pred = compare_input_with_cleaned_prediction(sample_input, prediction)
    input_present = check_input_in_prediction(sample_input, prediction)

    pred_controllers = extract_controllers(prediction)
    gt_controllers = extract_controllers(gt)

    tp = pred_controllers & gt_controllers
    fp = pred_controllers - gt_controllers
    fn = gt_controllers - pred_controllers

    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0.0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "match": match,
        "input_present": input_present,
        "tp": sorted(tp),
        "fp": sorted(fp),
        "fn": sorted(fn),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "pred_controllers": sorted(pred_controllers),
        "gt_controllers": sorted(gt_controllers),
    }


def evaluate_all(samples):
    results = []
    match_count = 0
    for i, (inp, pred, gt) in enumerate(samples):
        res = evaluate_prediction(inp, pred, gt)
        if res["match"]:
            match_count += 1
        results.append(res)

    accuracy = match_count / len(samples)
    print(f"\n✅ Cleaned Match Accuracy: {accuracy*100:.2f}% ({match_count}/{len(samples)})")
    return results


sample_input = "(raw)(hex){1}(v)(mix)<&|(raw)(v)(mix)&<&|(raw)&||(r)[{bout}(v)(prod)]{tout}(v)(pp)(v)(prod)n|(raw)(splt)[(hex){1}(mix)<1(prod)](v)1"
prediction = "(raw)(hex){1}(C){TC}_1(C){FC}_2(v)<_2(mix)<&|(raw)(C){FFC}_3<_4(v)<_3(mix)&<&|(raw)(C){FT}&_4||(r)[(C){TI}][(C){LC}_5][{bout}(v)<_5(prod)]{tout}(C){PC}_6(v)<_6(pp)[(C){M}](C){PI}(C){FC}_7(v)<_7(prod)n|(raw)(splt)[(hex){1}(mix)<1(prod)](v)1<_1"  
  
ground_truth = "(raw)(hex){1}(C){TC}_1(C){FC}_2(v)<_2(mix)<&|(raw)(C){FFC}_3<_4(v)<_3(mix)&<&|(raw)(C){FT}&_4||(r)[(C){TI}][(C){LC}_5][{bout}(v)<_5(prod)]{tout}(C){PC}_6(v)<_6(pp)[(C){M}](C){PI}(C){FC}_7(v)<_7(prod)n|(raw)(splt)[(hex){1}(mix)<1(prod)](v)1<_1"  
  
samples = [(sample_input, prediction, ground_truth)]

results = evaluate_all(samples)
match, cleaned_input, cleaned_pred = compare_input_with_cleaned_prediction(sample_input, prediction)
print("Cleaned Input:\n", cleaned_input)
print("Cleaned Prediction:\n", cleaned_pred)
# Pretty print the result
import pprint
pprint.pprint(results[0])
for i in range()

