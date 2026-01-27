import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
import re
import json
import time
from project_config.config import cfg
import string


def _normalize_answer_robustly(s: str) -> str:
    s = str(s).lower().strip()
    comma_number_pattern = "[-+]?\\d{1,3}(?:,\\d{3})+(?:\\.\\d+)?"
    comma_numbers = re.findall(comma_number_pattern, s)
    if comma_numbers:
        for comma_num in comma_numbers:
            cleaned_num = comma_num.replace(",", "")
            if cleaned_num.endswith(".0"):
                cleaned_num = cleaned_num[:-2]
            s = s.replace(comma_num, cleaned_num)
    numbers = re.findall("[-+]?\\d*\\.\\d+|\\d+", s)
    if len(numbers) == 1:
        non_numeric_part = re.sub("[-+]?\\d*\\.\\d+|\\d+", "", s).strip()
        if len(non_numeric_part.split()) <= 1:
            cleaned_num = numbers[0]
            if cleaned_num.endswith(".0"):
                cleaned_num = cleaned_num[:-2]
            return cleaned_num
    s = re.sub("\\b(a|an|the)\\b", " ", s)
    s = "".join((ch for ch in s if ch not in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{}~"))
    s = " ".join(s.split())
    return s


def _fallback_containment_check(model_output: str, processed_gold_list: list) -> bool:
    normalized_model_output = _normalize_answer_robustly(model_output)
    for gold_item in processed_gold_list:
        normalized_gold_item = _normalize_answer_robustly(gold_item)
        if re.fullmatch("[-+]?\\d*\\.\\d+|\\d+", normalized_gold_item):
            if not re.search(
                f"\\b{re.escape(normalized_gold_item)}\\b", normalized_model_output
            ):
                return False
        elif normalized_gold_item not in normalized_model_output:
            return False
    return True


def gumbel_softmax(logits, temperature=1.0, hard=False, eps=1e-10):
    """Gumbel-softmax for differentiable discrete routing during training."""
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )
    gumbels = (logits + gumbels) / (temperature + eps)
    y_soft = F.softmax(gumbels, dim=-1)
    if hard:
        index = y_soft.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, device=logits.device).scatter_(-1, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


def measure_expert_latency(
    expert_model, expert_name, get_intermediate_state_fn, num_warmup=5, num_repeats=20
):
    print(f"INFO: Measuring latency for {expert_name}.continue_generation_from_state")
    device = expert_model.device
    generation_params = cfg.get("GENERATION", {})
    for i in range(num_warmup):
        try:
            intermediate_state = get_intermediate_state_fn()
            if intermediate_state is None:
                print(f"WARN: Warmup intermediate state unavailable for {expert_name}; skipping warmup run.")
                continue
            _ = expert_model.continue_generation_from_state(
                intermediate_state, **generation_params
            )
        except Exception as e:
            print(f"WARN: Warmup error for {expert_name}: {e}")
            pass
    total_time = 0.0
    actual_repeats = 0
    for i in range(num_repeats):
        try:
            intermediate_state = get_intermediate_state_fn()
            if intermediate_state is None:
                print(f"ERROR: Missing intermediate state for {expert_name} at iteration {i}; skipping measurement.")
                continue
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = expert_model.continue_generation_from_state(
                intermediate_state, **generation_params
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            total_time += end_time - start_time
            actual_repeats += 1
        except Exception as e:
            print(f"ERROR: Timed run failed for {expert_name} at iteration {i}: {e}")
            import traceback

            traceback.print_exc()
            continue
    if actual_repeats == 0:
        print(f"ERROR: All timed runs failed for {expert_name}; returning default latency of 10.0 seconds.")
        return 10.0
    avg_latency = total_time / actual_repeats
    print(
        f"INFO: Average latency for {expert_name}.continue_generation_from_state over {actual_repeats} runs: {avg_latency:.4f} seconds"
    )
    return avg_latency


def calculate_expert_costs(expert_module_list_with_fusion_path_cost_first):
    costs = []
    for expert_or_path_obj in expert_module_list_with_fusion_path_cost_first:
        if hasattr(expert_or_path_obj, "cost"):
            costs.append(expert_or_path_obj.cost)
        elif isinstance(expert_or_path_obj, (float, int)):
            costs.append(expert_or_path_obj)
        else:
            print(
                f"WARN: No .cost attr for {type(expert_or_path_obj).__name__}. Defaulting to 10.0."
            )
            costs.append(10.0)
    print(f"INFO: Path costs for regularization: {costs}")
    return torch.tensor(costs, dtype=torch.float32)


def _normalize(txt: str) -> str:
    tbl = str.maketrans("", "", string.punctuation)
    return re.sub("\\s+", " ", txt.translate(tbl)).strip().lower()


NUM_PAT = re.compile("[-+]?\\d[\\d,]*\\.?\\d*")


def _clean_num(s: str) -> str:
    s = s.strip().rstrip(".,!?")
    return s.replace(",", "").lstrip("+").lstrip("0") or "0"


def extract_tqa_answer_list(model_output: str):
    if not isinstance(model_output, str) or not model_output:
        return []
    answer_match = re.search(
        '"answer"\\s*:\\s*\\[([\\s\\S]*?)\\]', model_output, re.DOTALL
    )
    if answer_match:
        content_str = answer_match.group(1)
        try:
            list_str = f"[{content_str}]"
            parsed_list = json.loads(list_str)
            if isinstance(parsed_list, list):
                return [str(item) for item in parsed_list]
        except json.JSONDecodeError:
            items = content_str.split(",")
            cleaned_items = [item.strip().strip("\"'").strip() for item in items]
            final_items = [item for item in cleaned_items if item]
            if final_items:
                return final_items
    m_out = model_output.replace("\n", " ").strip()
    pat = re.search("\\banswer(?:s| is|:)\\s*([A-Za-z0-9 ,./-]+)", m_out, flags=re.I)
    if pat:
        bits = [b.strip() for b in re.split(",|\\band\\b", pat.group(1)) if b.strip()]
        if bits:
            return bits
    nums = [_clean_num(n) for n in NUM_PAT.findall(m_out) if n.strip()]
    if len(set(nums)) == 1:
        return [nums[0]]
    return [m_out.rstrip(".,!?").strip()]


def parse_answer(answer):
    try:
        evaluated = eval(answer)
        if isinstance(evaluated, (int, float)):
            return float(evaluated)
        return answer
    except (NameError, SyntaxError, TypeError):
        return answer


def normalize_answer_for_em(s):
    def remove_articles(text):
        return re.sub("\\b(a|an|the)\\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        return "".join((ch for ch in text if ch not in exclude))

    def lower(text):
        return text.lower()

    if not isinstance(s, str):
        try:
            s_list = json.loads(s)
            if isinstance(s_list, list):
                s = " ".join(sorted([str(item) for item in s_list]))
            else:
                s = str(s_list)
        except (json.JSONDecodeError, TypeError):
            if isinstance(s, list):
                s = " ".join(sorted([str(item) for item in s]))
            else:
                s = str(s)
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def calculate_exact_match(prediction_str, target_str):
    return (
        1.0
        if normalize_answer_for_em(prediction_str)
        == normalize_answer_for_em(target_str)
        else 0.0
    )


def calculate_tabfact_accuracy(prediction_str, target_str):
    predicted_answers = extract_tqa_answer_list(prediction_str)
    if not predicted_answers:
        return 0.0
    predicted_text = (
        predicted_answers[0].lower()
        if isinstance(predicted_answers[0], str)
        else str(predicted_answers[0]).lower()
    )
    pred_bool = None
    if predicted_text in ["true", "correct", "yes", "t", "1", "entail"]:
        pred_bool = True
    elif predicted_text in ["false", "incorrect", "no", "f", "0", "contradict"]:
        pred_bool = False
    elif "not" in predicted_text and (
        "correct" in predicted_text or "true" in predicted_text
    ):
        pred_bool = False
    elif "correct" in predicted_text or "true" in predicted_text:
        pred_bool = True
    elif "incorrect" in predicted_text or "false" in predicted_text:
        pred_bool = False
    else:
        print(
            f"WARN: Cannot determine boolean from '{predicted_text}', defaulting to False"
        )
        pred_bool = False
    target_text = (
        target_str[0].lower()
        if isinstance(target_str, list)
        else str(target_str).lower()
    )
    target_bool = target_text in ["true", "correct", "yes", "t", "1", "entail"]
    return 1.0 if pred_bool == target_bool else 0.0


def calculate_infotabs_accuracy(prediction_str, target_str):
    predicted_answers = extract_tqa_answer_list(prediction_str)
    if not predicted_answers:
        return 0.0
    predicted_text = (
        predicted_answers[0].lower()
        if isinstance(predicted_answers[0], str)
        else str(predicted_answers[0]).lower()
    )
    if predicted_text in [
        "entail",
        "entailment",
        "entailment.",
        "entailed",
        "yes",
        "agree",
        "agrees",
        "supported",
        "support",
        "true",
    ]:
        pred_class = "entail"
    elif predicted_text in [
        "neutral",
        "neutral.",
        "neither",
        "unknown",
        "unclear",
        "can't determine",
        "cant determine",
        "not enough information",
    ]:
        pred_class = "neutral"
    elif predicted_text in [
        "contradict",
        "contradiction",
        "contradicts",
        "contradicted",
        "no",
        "disagree",
        "disagrees",
        "false",
        "wrong",
        "incorrect",
    ]:
        pred_class = "contradict"
    elif (
        "entail" in predicted_text
        or "agree" in predicted_text
        or "support" in predicted_text
    ):
        pred_class = "entail"
    elif "neutral" in predicted_text or "neither" in predicted_text:
        pred_class = "neutral"
    elif "contradict" in predicted_text or "disagree" in predicted_text:
        pred_class = "contradict"
    else:
        print(
            f"WARN: Cannot determine class from '{predicted_text}', defaulting to neutral"
        )
        pred_class = "neutral"
    if isinstance(target_str, list):
        target_text = target_str[0].lower()
    else:
        target_text = str(target_str).lower()
    if target_text in ["entail", "entailment", "entailed", "yes", "agree", "agrees"]:
        target_class = "entail"
    elif target_text in ["neutral", "neither"]:
        target_class = "neutral"
    elif target_text in [
        "contradict",
        "contradiction",
        "contradicts",
        "contradicted",
        "no",
        "disagree",
        "disagrees",
    ]:
        target_class = "contradict"
    else:
        target_class = target_text
    return 1.0 if pred_class == target_class else 0.0


def calculate_tqa_accuracy(prediction_str, target_str):
    predicted_answers = extract_tqa_answer_list(prediction_str)
    if isinstance(target_str, list):
        gold_answers = target_str
    else:
        try:
            parsed_target = json.loads(target_str)
            if isinstance(parsed_target, list):
                gold_answers = parsed_target
            else:
                gold_answers = [str(target_str)]
        except (json.JSONDecodeError, TypeError):
            gold_answers = [str(target_str)]
    normalized_predicted = sorted(
        [normalize_answer_for_em(ans) for ans in predicted_answers]
    )
    normalized_gold = sorted([normalize_answer_for_em(ans) for ans in gold_answers])
    return 1.0 if normalized_predicted == normalized_gold else 0.0


def calculate_hitab_accuracy(prediction_str, target_str):
    predicted_answers = extract_tqa_answer_list(prediction_str)
    if not predicted_answers:
        return 0.0
    target_answers = []
    if isinstance(target_str, str):
        try:
            parsed = json.loads(target_str)
            if isinstance(parsed, list):
                target_answers = parsed
            else:
                target_answers = [target_str]
        except json.JSONDecodeError:
            target_answers = [target_str]
    elif isinstance(target_str, list):
        target_answers = target_str
    else:
        target_answers = [str(target_str)]
    predicted_answer_list = [parse_answer(ans) for ans in predicted_answers]
    gold_answer_list = [parse_answer(ans) for ans in target_answers]
    if set(gold_answer_list) == set(predicted_answer_list):
        return 1.0
    return 0.0


def calculate_bleu_for_fetaqa(prediction_str, target_str):
    if not prediction_str or not target_str:
        return 0.0
    references = [[str(target_str)]]
    predictions = [str(prediction_str)]
    bleu = BLEU(max_ngram_order=4)
    score = bleu.corpus_score(predictions, references)
    return score.score / 100.0


def _get_question_text(qid: str, id2item: dict) -> str:
    it = id2item.get(qid, {})
    for k in (
        "original_query",
        "question",
        "question_text",
        "statement",
        "prompt",
        "input",
    ):
        if k in it and it[k]:
            text = str(it[k]).strip()
            if k == "input" or text.lower().startswith("problem:"):
                text = text.split("\n", 1)[0]
                text = re.sub("^\\s*problem:\\s*", "", text, flags=re.I)
            return text
    return "<question text missing>"


def _get_sorted_groups(groups_dict):
    def sort_key(group_name):
        if group_name == "Fusion":
            return 0
        elif group_name == "VLMExpert":
            return 1
        elif group_name == "TextExpert":
            return 2
        else:
            return 3

    return sorted(groups_dict.items(), key=lambda x: sort_key(x[0]))


def _extract_first_sentence_for_fetaqa(text):
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        import os

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        nltk_data_path = os.path.join(project_root, "nltk_data")
        os.makedirs(nltk_data_path, exist_ok=True)
        if nltk_data_path not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_path)
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            print("INFO: Downloading NLTK punkt tokenizer")
            nltk.download("punkt", download_dir=nltk_data_path, quiet=True)
        text = text.strip()
        if not text:
            return text
        sentences = sent_tokenize(text)
        if sentences and sentences[0].strip():
            return sentences[0].strip()
        else:
            return text
    except ImportError:
        try:
            import spacy

            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                nlp = spacy.blank("en")
                nlp.add_pipe("sentencizer")
            text = text.strip()
            if not text:
                return text
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
            if sentences and sentences[0]:
                return sentences[0]
            else:
                return text
        except ImportError:
            text = text.strip()
            if not text:
                return text
            import re

            abbreviations = [
                "Mr",
                "Mrs",
                "Ms",
                "Dr",
                "Prof",
                "Jr",
                "Sr",
                "vs",
                "etc",
                "Inc",
                "Ltd",
                "Co",
            ]
            temp_text = text
            for abbr in abbreviations:
                temp_text = re.sub(
                    f"\\b{abbr}\\.", f"{abbr}__DOT__", temp_text, flags=re.IGNORECASE
                )
            sentences = re.split("[.!?]+\\s+", temp_text)
            if sentences and sentences[0].strip():
                first_sentence = sentences[0].replace("__DOT__", ".")
                if first_sentence and (not first_sentence[-1] in ".!?"):
                    first_sentence += "."
                return first_sentence.strip()
            return text


def _extract_fetaqa_answer_for_evaluation(model_output, selected_expert):
    if not isinstance(model_output, str) or not model_output:
        return ""
    cleaned_output = model_output.strip()
    if selected_expert == "TextExpert":
        return _extract_first_sentence_for_fetaqa(cleaned_output)
    else:
        return cleaned_output


def get_accuracy_for_training(prediction_str: str, gold_answer_list_or_str) -> float:
    try:
        pred_list = extract_tqa_answer_list(prediction_str)
        if isinstance(gold_answer_list_or_str, list):
            raw_gold_list = gold_answer_list_or_str
        else:
            try:
                parsed_gold = json.loads(gold_answer_list_or_str)
                if isinstance(parsed_gold, list):
                    raw_gold_list = parsed_gold
                else:
                    raw_gold_list = [str(gold_answer_list_or_str)]
            except (json.JSONDecodeError, TypeError):
                raw_gold_list = [str(gold_answer_list_or_str)]
        processed_gold_list = []
        for gold in raw_gold_list:
            processed_gold_list.extend(str(gold).split("|"))
        pred_set = {_normalize_answer_robustly(p) for p in pred_list}
        gold_set = {_normalize_answer_robustly(g) for g in processed_gold_list}
        is_correct = gold_set == pred_set
        if not is_correct:
            is_correct = _fallback_containment_check(
                prediction_str, processed_gold_list
            )
        return 1.0 if is_correct else 0.0
    except Exception as e:
        return 0.0
