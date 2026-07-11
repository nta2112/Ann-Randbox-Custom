import logging
from collections import OrderedDict, defaultdict

import numpy as np
import torch

import detectron2.utils.comm as comm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator


def _bbox_iou_xyxy(box, boxes):
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)

    ixmin = np.maximum(box[0], boxes[:, 0])
    iymin = np.maximum(box[1], boxes[:, 1])
    ixmax = np.minimum(box[2], boxes[:, 2])
    iymax = np.minimum(box[3], boxes[:, 3])
    iw = np.maximum(ixmax - ixmin, 0.0)
    ih = np.maximum(iymax - iymin, 0.0)
    inter = iw * ih

    box_area = np.maximum(box[2] - box[0], 0.0) * np.maximum(box[3] - box[1], 0.0)
    boxes_area = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
    union = box_area + boxes_area - inter
    return inter / np.maximum(union, np.finfo(np.float32).eps)


def _voc_ap(recalls, precisions):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    changing_points = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1])


class OpenWorldCOCOEvaluator(DatasetEvaluator):
    """
    COCO-format evaluator for the open-world summary metrics used by RandBox.

    It reports VOC-style AP@50, final precision, and final recall for previous,
    current, known, and unknown class groups. The regular Detectron2 COCOEvaluator
    should still be used for the standard COCO AP/AP50/AP75 metrics.
    """

    def __init__(self, dataset_name, cfg, iou_threshold=0.5, score_threshold=0.0):
        self._dataset_name = dataset_name
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._metadata = MetadataCatalog.get(dataset_name)
        self._dataset_dicts = DatasetCatalog.get(dataset_name)
        self._iou_threshold = iou_threshold
        self._score_threshold = score_threshold

        self.prev_intro_cls = cfg.TEST.PREV_INTRODUCED_CLS
        self.curr_intro_cls = cfg.TEST.CUR_INTRODUCED_CLS
        self.total_num_class = cfg.MODEL.RandBox.NUM_CLASSES
        self.num_seen_classes = self.prev_intro_cls + self.curr_intro_cls

        self._class_names = list(getattr(self._metadata, "thing_classes", []))
        self.unknown_class_index = self.total_num_class - 1
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            self.unknown_class_index = self._metadata.thing_dataset_id_to_contiguous_id.get(
                self.unknown_class_index,
                self.unknown_class_index,
            )
        if self._class_names and "unknown" in self._class_names:
            self.unknown_class_index = self._class_names.index("unknown")

        self._logger.info("OpenWorldCOCOEvaluator: unknown_class_index=%s", self.unknown_class_index)

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input_per_image, output_per_image in zip(inputs, outputs):
            image_id = input_per_image["image_id"]
            instances = output_per_image["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()

            keep = scores >= self._score_threshold
            for box, score, cls in zip(boxes[keep], scores[keep], classes[keep]):
                if cls == -100:
                    continue
                self._predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": int(cls),
                        "bbox": box.astype(np.float32),
                        "score": float(score),
                    }
                )

    def evaluate(self):
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return

        predictions = []
        for predictions_per_rank in all_predictions:
            predictions.extend(predictions_per_rank)

        gt_by_class = self._build_ground_truth_by_class()
        pred_by_class = defaultdict(list)
        for pred in predictions:
            pred_by_class[pred["category_id"]].append(pred)

        # Debug: count predictions per class and unknown predictions
        pred_counts = {class_id: len(preds) for class_id, preds in pred_by_class.items()}
        unknown_preds = len(pred_by_class.get(self.unknown_class_index, []))
        gt_unknown = len(gt_by_class.get(self.unknown_class_index, {}))
        self._logger.info(
            "Open-world COCO pred counts: unknown=%s, total_classes_with_preds=%s", unknown_preds, len(pred_counts)
        )
        self._logger.info("Open-world COCO unknown GT images: %s", gt_unknown)
        if unknown_preds == 0:
            top_pred_classes = sorted(pred_counts.items(), key=lambda x: -x[1])[:10]
            self._logger.warning(
                "No predictions for unknown class %s. Top predicted classes: %s. Unknown metrics will be 0 unless model predicts class %s.",
                self.unknown_class_index,
                top_pred_classes,
                self.unknown_class_index,
            )

        class_metrics = {}
        for class_id in range(self.total_num_class):
            class_metrics[class_id] = self._evaluate_class(
                gt_by_class.get(class_id, {}),
                pred_by_class.get(class_id, []),
            )

        groups = OrderedDict(
            [
                ("previous_known", range(0, self.prev_intro_cls)),
                ("current_known", range(self.prev_intro_cls, self.num_seen_classes)),
                ("known", range(0, self.num_seen_classes)),
                ("unknown", [self.unknown_class_index]),
            ]
        )

        ret = OrderedDict()
        ret["open_world"] = OrderedDict()
        for group_name, class_ids in groups.items():
            summary = self._summarize_group(class_metrics, list(class_ids))
            ret["open_world"][f"{group_name}_mAP50"] = summary["mAP50"]
            ret["open_world"][f"{group_name}_precision50"] = summary["precision50"]
            ret["open_world"][f"{group_name}_recall50"] = summary["recall50"]

        self._logger.info("Open-world COCO metrics: %s", ret["open_world"])
        print("Open-world COCO metrics:", ret["open_world"])
        return ret

    def _build_ground_truth_by_class(self):
        gt_by_class = defaultdict(lambda: defaultdict(list))
        for dataset_dict in self._dataset_dicts:
            image_id = dataset_dict["image_id"]
            for ann in dataset_dict.get("annotations", []):
                class_id = ann["category_id"]
                bbox = ann["bbox"]
                if ann.get("bbox_mode") is not None:
                    from detectron2.structures import BoxMode

                    bbox = BoxMode.convert(bbox, ann["bbox_mode"], BoxMode.XYXY_ABS)
                else:
                    x, y, w, h = bbox
                    bbox = [x, y, x + w, y + h]
                gt_by_class[int(class_id)][image_id].append(np.asarray(bbox, dtype=np.float32))
        return gt_by_class

    def _evaluate_class(self, gt_for_class, predictions):
        npos = sum(len(boxes) for boxes in gt_for_class.values())
        if npos == 0:
            return {"AP50": 0.0, "precision50": 0.0, "recall50": 0.0, "num_gt": 0}

        predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)
        detected = {image_id: np.zeros(len(boxes), dtype=bool) for image_id, boxes in gt_for_class.items()}
        tp = np.zeros(len(predictions), dtype=np.float32)
        fp = np.zeros(len(predictions), dtype=np.float32)

        for idx, pred in enumerate(predictions):
            image_id = pred["image_id"]
            gt_boxes = np.asarray(gt_for_class.get(image_id, []), dtype=np.float32)
            if gt_boxes.size == 0:
                fp[idx] = 1.0
                continue

            overlaps = _bbox_iou_xyxy(pred["bbox"], gt_boxes)
            best_match = int(np.argmax(overlaps))
            if overlaps[best_match] >= self._iou_threshold and not detected[image_id][best_match]:
                tp[idx] = 1.0
                detected[image_id][best_match] = True
            else:
                fp[idx] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / float(npos)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float32).eps)

        return {
            "AP50": float(_voc_ap(recalls, precisions) * 100.0) if len(predictions) else 0.0,
            "precision50": float(precisions[-1] * 100.0) if len(predictions) else 0.0,
            "recall50": float(recalls[-1] * 100.0) if len(predictions) else 0.0,
            "num_gt": npos,
        }

    @staticmethod
    def _summarize_group(class_metrics, class_ids):
        valid_metrics = [class_metrics[class_id] for class_id in class_ids if class_metrics[class_id]["num_gt"] > 0]
        if not valid_metrics:
            return {"mAP50": 0.0, "precision50": 0.0, "recall50": 0.0}
        return {
            "mAP50": float(np.mean([metric["AP50"] for metric in valid_metrics])),
            "precision50": float(np.mean([metric["precision50"] for metric in valid_metrics])),
            "recall50": float(np.mean([metric["recall50"] for metric in valid_metrics])),
        }
