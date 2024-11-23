from .kp20k_evaluator import evaluate_kp20k
from functools import partial
import os
import logging
from ..utils import Timer

logger = logging.getLogger()


# Evaluation Script
def evaluate_script(
    cfg, candidate, stats, mode, evaluate_method, metric_name="max_f1_score", max_k=100
):
    logger.info("*" * 80)
    logger.info("Start Evaluating : Mode = %s || Epoch = %d" % (mode, stats["epoch"]))
    epoch_time = Timer()

    evaluated_K = ["O"]
    evaluated_K.extend([x for x in range(1, 10) if x <= max_k])
    evaluated_K.extend([x for x in [10, 15] if x <= max_k])
    evaluated_K.extend([x for x in range(20, max_k + 1, 10) if x <= max_k])

    reference_filename = os.path.join(cfg.dir.data, cfg.data.dataset, f"{mode}.json")
    f1_scores, precision_scores, recall_scores, category_f1_scores = evaluate_method(
        cfg, candidate, reference_filename
    )

    precision_str = "Precision@K: "
    recall_str = "Recall@K: "
    f1_scores_str = "F1@K: "

    for i in precision_scores:
        if i in evaluated_K:
            precision_str += f"{precision_scores[i]:.4f} @{i} *** "
            recall_str += f"{recall_scores[i]:.4f} @{i} *** "
            f1_scores_str += f"{f1_scores[i]:.4f} @{i} *** "

    logger.info(precision_str)
    logger.info(recall_str)
    logger.info("*" * 60)

    categories_f1 = list(category_f1_scores.values())[0].keys()

    for category_f1 in categories_f1:
        category_f1_str = f"Category{' > ' if isinstance(category_f1, int) else ' '}{category_f1} with F1@K: "
        for i in precision_scores:
            if i in evaluated_K:
                category_f1_str += f"{category_f1_scores[i][category_f1]:.4f} @{i} *** "
        logger.info(category_f1_str)
        logger.info("*" * 10)

    logger.info("*" * 60)
    logger.info(f1_scores_str)
    logger.info("*" * 60)

    f1s_at_ks = {i: f1_scores[i] for i in f1_scores if i != "O"}
    best_k = max(f1s_at_ks, key=f1s_at_ks.get)
    logger.info(f"Best F1@B:  {f1s_at_ks[best_k]:.4f}@{best_k}")

    best_f1 = f1_scores["O"]
    if best_f1 > stats[metric_name]:
        stats[metric_name] = best_f1

    logger.info(
        "End Evaluating : Mode = %s || Epoch = %d (Time: %.2f (s)) "
        % (mode, stats["epoch"], epoch_time.time())
    )
    logger.info("*" * 80)
    return stats


# -------------------------------------------------------------------------------------------
# Select Evaluation Scripts
# -------------------------------------------------------------------------------------------


def select_eval_script():
    return partial(evaluate_script, evaluate_method=evaluate_kp20k), "max_f1_score"
