import torch
from torch import nn
import os
from ..utils import concat_box_prediction_layers
from atss_core.layers import SigmoidFocalLoss
from atss_core.structures.boxlist_ops import boxlist_iou
from atss_core.structures.boxlist_ops import cat_boxlist
from atss_core.structures.bounding_box import BoxList

INF = 100000000


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


class ATSSLossComputation(object):

    def __init__(self, cfg, box_coder):
        self.cfg = cfg
        self.cls_loss_func = SigmoidFocalLoss(cfg.MODEL.ATSS.LOSS_GAMMA, cfg.MODEL.ATSS.LOSS_ALPHA)
        self.iou_pred_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.box_coder = box_coder

    def GIoULoss(self, pred, target, anchor, weight=None):
        pred_boxes = self.box_coder.decode(pred.view(-1, 4), anchor.view(-1, 4))
        pred_x1 = pred_boxes[:, 0]
        pred_y1 = pred_boxes[:, 1]
        pred_x2 = pred_boxes[:, 2]
        pred_y2 = pred_boxes[:, 3]
        pred_x2 = torch.max(pred_x1, pred_x2)
        pred_y2 = torch.max(pred_y1, pred_y2)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

        gt_boxes = self.box_coder.decode(target.view(-1, 4), anchor.view(-1, 4))
        target_x1 = gt_boxes[:, 0]
        target_y1 = gt_boxes[:, 1]
        target_x2 = gt_boxes[:, 2]
        target_y2 = gt_boxes[:, 3]
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

        x1_intersect = torch.max(pred_x1, target_x1)
        y1_intersect = torch.max(pred_y1, target_y1)
        x2_intersect = torch.min(pred_x2, target_x2)
        y2_intersect = torch.min(pred_y2, target_y2)
        area_intersect = torch.zeros(pred_x1.size()).to(pred)
        mask = (y2_intersect > y1_intersect) * (x2_intersect > x1_intersect)
        area_intersect[mask] = (x2_intersect[mask] - x1_intersect[mask]) * (y2_intersect[mask] - y1_intersect[mask])

        x1_enclosing = torch.min(pred_x1, target_x1)
        y1_enclosing = torch.min(pred_y1, target_y1)
        x2_enclosing = torch.max(pred_x2, target_x2)
        y2_enclosing = torch.max(pred_y2, target_y2)
        area_enclosing = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing) + 1e-7

        area_union = pred_area + target_area - area_intersect + 1e-7
        ious = area_intersect / area_union
        gious = ious - (area_enclosing - area_union) / area_enclosing

        losses = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()

    def prepare_targets(self, targets, anchors, regressed_boxes, classif_results):
        cls_labels = []
        reg_labels = []
        reg_targets = []
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes_per_im = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")

            anchors_per_im = cat_boxlist(anchors[im_i])
            regressed_per_im = cat_boxlist(regressed_boxes[im_i])
            classif_per_im = torch.cat(classif_results[im_i], 0)
            num_gt = bboxes_per_im.shape[0]
            
            gt_cx = (bboxes_per_im[:, 2] + bboxes_per_im[:, 0]) / 2.0
            gt_cy = (bboxes_per_im[:, 3] + bboxes_per_im[:, 1]) / 2.0
            gt_points = torch.stack((gt_cx, gt_cy), dim=1)

            anchors_cx_per_im = (anchors_per_im.bbox[:, 2] + anchors_per_im.bbox[:, 0]) / 2.0
            anchors_cy_per_im = (anchors_per_im.bbox[:, 3] + anchors_per_im.bbox[:, 1]) / 2.0
            anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)

            distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()

            # Selecting candidates based on the center distance between anchor box and object
            num_anchors_per_loc = len(self.cfg.MODEL.ATSS.ASPECT_RATIOS) * self.cfg.MODEL.ATSS.SCALES_PER_OCTAVE
            num_anchors_per_level = [len(anchors_per_level.bbox) for anchors_per_level in anchors[im_i]]
            candidate_idxs = []
            star_idx = 0
            for level, anchors_per_level in enumerate(anchors[im_i]):
                end_idx = star_idx + num_anchors_per_level[level]
                distances_per_level = distances[star_idx:end_idx, :]
                topk = min(self.cfg.MODEL.ATSS.TOPK * num_anchors_per_loc, num_anchors_per_level[level])
                _, topk_idxs_per_level = distances_per_level.topk(topk, dim=0, largest=False)
                candidate_idxs.append(topk_idxs_per_level + star_idx)
                star_idx = end_idx
            candidate_idxs = torch.cat(candidate_idxs, dim=0)

            ### Loc to Cls ###
            loc2cls_candidate_idxs = candidate_idxs.clone()
            # Using the sum of mean and standard deviation as the IoU threshold to select final positive samples
            ious = boxlist_iou(regressed_per_im, targets_per_im)
            candidate_ious = ious[loc2cls_candidate_idxs, torch.arange(num_gt)]
            iou_mean_per_gt = candidate_ious.mean(0)
            iou_std_per_gt = candidate_ious.std(0)
            iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
            is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

            # Limiting the final positive samples’ center to object
            anchor_num = anchors_cx_per_im.shape[0]
            for ng in range(num_gt):
                loc2cls_candidate_idxs[:, ng] += ng * anchor_num
            e_anchors_cx = anchors_cx_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
            e_anchors_cy = anchors_cy_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
            loc2cls_candidate_idxs = loc2cls_candidate_idxs.view(-1)
            l = e_anchors_cx[loc2cls_candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 0]
            t = e_anchors_cy[loc2cls_candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 1]
            r = bboxes_per_im[:, 2] - e_anchors_cx[loc2cls_candidate_idxs].view(-1, num_gt)
            b = bboxes_per_im[:, 3] - e_anchors_cy[loc2cls_candidate_idxs].view(-1, num_gt)
            is_in_gts = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
            is_pos = is_pos & is_in_gts

            # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
            ious_inf = torch.full_like(ious, -INF).t().contiguous().view(-1)
            index = loc2cls_candidate_idxs.view(-1)[is_pos.view(-1)]
            ious_inf[index] = ious.t().contiguous().view(-1)[index]
            ious_inf = ious_inf.view(num_gt, -1).t()

            anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)
            cls_labels_per_im = labels_per_im[anchors_to_gt_indexs]
            cls_labels_per_im[anchors_to_gt_values == -INF] = 0

            ### Cls to Loc ###
            cls2loc_candidate_idxs = candidate_idxs.clone()
            # Using the sum of mean and standard deviation as the IoU threshold to select final positive samples
            sigma = 2.0
            ious = boxlist_iou(anchors_per_im, targets_per_im) ** ((sigma-classif_per_im[:, labels_per_im-1])/sigma)
            candidate_ious = ious[cls2loc_candidate_idxs, torch.arange(num_gt)]
            iou_mean_per_gt = candidate_ious.mean(0)
            iou_std_per_gt = candidate_ious.std(0)
            iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
            is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

            # Limiting the final positive samples’ center to object
            anchor_num = anchors_cx_per_im.shape[0]
            for ng in range(num_gt):
                cls2loc_candidate_idxs[:, ng] += ng * anchor_num
            e_anchors_cx = anchors_cx_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
            e_anchors_cy = anchors_cy_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
            cls2loc_candidate_idxs = cls2loc_candidate_idxs.view(-1)
            l = e_anchors_cx[cls2loc_candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 0]
            t = e_anchors_cy[cls2loc_candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 1]
            r = bboxes_per_im[:, 2] - e_anchors_cx[cls2loc_candidate_idxs].view(-1, num_gt)
            b = bboxes_per_im[:, 3] - e_anchors_cy[cls2loc_candidate_idxs].view(-1, num_gt)
            is_in_gts = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
            is_pos = is_pos & is_in_gts

            # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
            ious_inf = torch.full_like(ious, -INF).t().contiguous().view(-1)
            index = cls2loc_candidate_idxs.view(-1)[is_pos.view(-1)]
            ious_inf[index] = ious.t().contiguous().view(-1)[index]
            ious_inf = ious_inf.view(num_gt, -1).t()

            anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)
            reg_labels_per_im = labels_per_im[anchors_to_gt_indexs]
            reg_labels_per_im[anchors_to_gt_values == -INF] = 0
            matched_gts = bboxes_per_im[anchors_to_gt_indexs]

            cls_labels.append(cls_labels_per_im)
            reg_labels.append(reg_labels_per_im)
            reg_targets.append(self.box_coder.encode(matched_gts, anchors_per_im.bbox))

        return cls_labels, reg_labels, reg_targets

    def compute_ious(self, boxes1, boxes2):
        area1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)
        area2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1)
        lt = torch.max(boxes1[:, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        wh = (rb - lt + 1).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        return inter / (area1 + area2 - inter)

    def __call__(self, box_cls, box_regression, iou_pred, targets, anchors):
        all_regressed_boxes = list()
        all_classif_results = list()
        num_scale = len(box_regression)
        num_batch = box_regression[0].size(0)
        num_classes = box_cls[0].size(1)
        for batch in range(num_batch):
            perimage_regressed_boxes = list()
            perimage_classif_results = list()
            for scale in range(num_scale):
                perimage_regressed_boxes.append(
                    BoxList(
                        self.box_coder.decode(
                            box_regression[scale][batch,:,:,:].clone().detach().permute(1, 2, 0).contiguous().view(-1, 4), anchors[batch][scale].bbox.view(-1, 4)
                            ),
                        anchors[batch][scale].size
                        )
                    )
                perimage_classif_results.append(
                    box_cls[scale][batch,:,:,:].clone().detach().sigmoid().permute(1, 2, 0).contiguous().view(-1, num_classes)
                    )
            all_regressed_boxes.append(perimage_regressed_boxes)
            all_classif_results.append(perimage_classif_results)

        cls_labels, reg_labels, reg_targets = self.prepare_targets(targets, anchors, all_regressed_boxes, all_classif_results)

        box_cls_flatten, box_regression_flatten = concat_box_prediction_layers(box_cls, box_regression)
        iou_pred_flatten = [ct.permute(0, 2, 3, 1).reshape(num_batch, -1, 1) for ct in iou_pred]
        iou_pred_flatten = torch.cat(iou_pred_flatten, dim=1).reshape(-1)

        cls_labels_flatten = torch.cat(cls_labels, dim=0)
        reg_labels_flatten = torch.cat(reg_labels, dim=0)
        reg_targets_flatten = torch.cat(reg_targets, dim=0)
        anchors_flatten = torch.cat([cat_boxlist(anchors_per_image).bbox for anchors_per_image in anchors], dim=0)

        num_gpus = get_num_gpus()

        cls_pos_inds = torch.nonzero(cls_labels_flatten > 0).squeeze(1)
        cls_num_pos_avg_per_gpu = max(reduce_sum(cls_pos_inds.new_tensor([cls_pos_inds.numel()])).item() / float(num_gpus), 1.0)
        cls_loss = self.cls_loss_func(box_cls_flatten, cls_labels_flatten.int()) / cls_num_pos_avg_per_gpu

        reg_pos_inds = torch.nonzero(reg_labels_flatten > 0).squeeze(1)
        reg_num_pos_avg_per_gpu = max(reduce_sum(reg_pos_inds.new_tensor([reg_pos_inds.numel()])).item() / float(num_gpus), 1.0)
        box_regression_flatten = box_regression_flatten[reg_pos_inds]
        reg_targets_flatten = reg_targets_flatten[reg_pos_inds]
        anchors_flatten = anchors_flatten[reg_pos_inds]
        iou_pred_flatten = iou_pred_flatten[reg_pos_inds]
        gt_boxes = self.box_coder.decode(reg_targets_flatten, anchors_flatten)
        boxes = self.box_coder.decode(box_regression_flatten, anchors_flatten).detach()
        iou_pred_targets = self.compute_ious(gt_boxes, boxes)
        sum_iou_pred_targets_avg_per_gpu = reduce_sum(iou_pred_targets.sum()).item() / float(num_gpus)
        if cls_pos_inds.numel() > 0:
            reg_loss = self.GIoULoss(box_regression_flatten, reg_targets_flatten, anchors_flatten, weight=iou_pred_targets) / sum_iou_pred_targets_avg_per_gpu
            iou_pred_loss = self.iou_pred_loss_func(iou_pred_flatten, iou_pred_targets) / reg_num_pos_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum()
            iou_pred_loss = iou_pred_flatten.sum()

        return cls_loss, reg_loss * self.cfg.MODEL.ATSS.REG_LOSS_WEIGHT, iou_pred_loss * 0.5


def make_atss_loss_evaluator(cfg, box_coder):
    loss_evaluator = ATSSLossComputation(cfg, box_coder)
    return loss_evaluator
