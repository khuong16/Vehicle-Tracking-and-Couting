# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import torch
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}

deepsort = None

# Clear global variables section
global object_counter, object_counter1, total_vehicles, processed_ids, cumulative_counts

# Initialize counters with default values for vehicle types
object_counter = {'car': 0, 'truck': 0, 'bus': 0, 'motorbike': 0}  # Out/South direction 
object_counter1 = {'car': 0, 'truck': 0, 'bus': 0, 'motorbike': 0}  # In/North direction

# Maintain cumulative counts of vehicles detected over time
cumulative_counts = {'car': 0, 'truck': 0, 'bus': 0, 'motorbike': 0}

# Set of unique object IDs to avoid double-counting
counted_object_ids = set()

# Dictionary to track unique vehicle IDs for total count
total_vehicles = {}

# Set to track IDs that have already crossed the line
processed_ids = set()

# Set to False for single lane mode, True for dual lane mode
dual_lane_mode = False

# Map YOLO/COCO class names to our standardized names
# This handles variations in class names returned by the model
vehicle_name_mapping = {
    'car': 'car',
    'truck': 'truck',
    'bus': 'bus',
    'motorbike': 'motorbike',
    'motorcycles': 'motorbike',  # Plural form from YOLO detections
    'motorcycle': 'motorbike',   # Singular form  
    'lorry': 'truck',            # Alternative name
    'automobile': 'car'          # Alternative name
}

# These are the only vehicle types we want to display
display_vehicle_types = ['car', 'truck', 'bus', 'motorbike']

line = [(100, 500), (1050, 500)]
def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
##########################################################################################
def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #person
        color = (85,45,255)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def get_direction(point1, point2):
    direction_str = ""

    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    else:
        direction_str += ""

    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""

    return direction_str
def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0), frame_counts={}):
    """
    Draw bounding boxes with tracking IDs and handle vehicle counting
    """
    global object_counter, object_counter1, total_vehicles, processed_ids, cumulative_counts, counted_object_ids
    
    # Draw the counting line
    cv2.line(img, line[0], line[1], (46, 162, 112), 3)

    height, width, _ = img.shape
    
    # Debug: Print the frame_counts to see what objects were detected
    print(f"Frame detections: {frame_counts}")
    
    # Remove tracked points from buffer if object is lost
    for key in list(data_deque):
        if identities is not None and key not in identities:
            data_deque.pop(key)

    # Update cumulative counts with new detections in this frame
    if len(bbox) > 0 and identities is not None:
        for i, box in enumerate(bbox):
            # Get object details
            obj_id = int(identities[i]) if identities is not None else 0
            obj_name = names[object_id[i]].lower()
            
            # Debug: Print the detected object and its mapped type
            vehicle_type = vehicle_name_mapping.get(obj_name)
            print(f"Detected: {obj_name} → Mapped to: {vehicle_type}")
            
            # Only count vehicle types we're interested in
            if vehicle_type in display_vehicle_types:
                # If we haven't counted this object ID before
                if obj_id not in counted_object_ids:
                    # Increment the cumulative count for this vehicle type
                    cumulative_counts[vehicle_type] += 1
                    # Mark this object as counted
                    counted_object_ids.add(obj_id)
                    # Update the total vehicles count
                    total_vehicles[obj_id] = vehicle_type
                    print(f"Added to count: {vehicle_type}, New total: {cumulative_counts[vehicle_type]}")
            
            # Continue with normal drawing and processing
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]

            # Calculate center point of the detection
            center = (int((x2+x1) / 2), int((y2+y1) / 2))

            # Create new buffer for new object
            if obj_id not in data_deque:  
                data_deque[obj_id] = deque(maxlen=64)

            # Get class details
            color = compute_color_for_labels(object_id[i])
            
            # Create label with ID and class
            label = f"{obj_id}:{obj_name}"
            
            # Add center to trajectory buffer
            data_deque[obj_id].appendleft(center)
            
            # Skip further processing if it's not a vehicle we're interested in
            if vehicle_type not in display_vehicle_types:
                UI_box(box, img, label=label, color=color, line_thickness=2)
                continue

            # Check if object has crossed the line for counting
            if len(data_deque[obj_id]) >= 2:
                # If this ID hasn't been processed yet for line crossing
                if obj_id not in processed_ids:
                    if intersect(data_deque[obj_id][0], data_deque[obj_id][1], line[0], line[1]):
                        # Mark the line as crossed
                        cv2.line(img, line[0], line[1], (255, 255, 255), 3)
                        
                        # Get movement direction
                        direction = get_direction(data_deque[obj_id][0], data_deque[obj_id][1])
                        
                        if dual_lane_mode:
                            # In dual lane mode, track direction
                            if "South" in direction:
                                object_counter[vehicle_type] += 1
                            if "North" in direction:
                                object_counter1[vehicle_type] += 1
                        else:
                            # In single lane mode, count all vehicles regardless of direction
                            object_counter[vehicle_type] += 1
                        
                        # Mark this ID as processed for line crossing
                        processed_ids.add(obj_id)
            
            # Draw box for this detection
            UI_box(box, img, label=label, color=color, line_thickness=2)
            
            # Draw trajectory lines
            for i in range(1, len(data_deque[obj_id])):
                if data_deque[obj_id][i - 1] is None or data_deque[obj_id][i] is None:
                    continue
                # Generate dynamic thickness for trail lines
                thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
                # Draw trail line
                cv2.line(img, data_deque[obj_id][i - 1], data_deque[obj_id][i], color, thickness)

    # Print current cumulative counts for debugging
    print(f"Cumulative counts: {cumulative_counts}")
    print(f"Total unique vehicles: {len(total_vehicles)}")

    # Background color for text displays
    bg_color = [255, 130, 0]  # Blue-orange color in BGR
    
    # Display the cumulative counts on the left side of the screen (instead of center)
    left_x = 20  # Starting position from left edge
    cv2.line(img, (left_x, 25), (left_x + 300, 25), bg_color, 40)
    cv2.putText(img, f'Cumulative Vehicle Counts', (left_x + 10, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
    
    # Display each vehicle type count
    idx = 0
    for vehicle_type in display_vehicle_types:
        count = cumulative_counts[vehicle_type]
        cnt_str = f"{vehicle_type}: {count}"
        # Background line positioned on left side
        cv2.line(img, (left_x, 65 + (idx*40)), (left_x + 200, 65 + (idx*40)), bg_color, 30)
        # Text positioned on left side
        cv2.putText(img, cnt_str, (left_x + 10, 75 + (idx*40)), 0, 1.2, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        idx += 1
    
    # Display total unique vehicle count at bottom-right
    total_count = len(total_vehicles)
    cv2.line(img, (width - 300, height - 40), (width, height - 40), bg_color, 40)
    cv2.putText(img, f'Count vehicle: {total_count}', (width - 290, height - 30), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)
    
    return img


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        all_outputs = []
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        all_outputs.append(det)
        
        # Create a frame-specific detection counter
        frame_counts = {}
        
        if len(det) > 0:
            # Count detections per class in this frame
            for c in det[:, 5].unique():
                class_idx = int(c)
                class_name = self.model.names[class_idx].lower()
                count = (det[:, 5] == c).sum().item()  # detections per class
                frame_counts[class_name] = count
                log_string += f"{count} {self.model.names[class_idx]}{'s' * (count > 1)}, "
        
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []
        for *xyxy, conf, cls in reversed(det):
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([conf.item()])
            oids.append(int(cls))
        
        if len(xywh_bboxs) > 0:
            xywhs = torch.Tensor(xywh_bboxs)
            confss = torch.Tensor(confs)
            
            outputs = deepsort.update(xywhs, confss, oids, im0)
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -2]
                object_id = outputs[:, -1]
                
                # Pass frame_counts to draw_boxes to display current frame detections
                draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities, frame_counts=frame_counts)
            else:
                # Even if no tracking, still show detection counts
                draw_boxes(im0, [], self.model.names, [], None, frame_counts=frame_counts)
        else:
            # No detections but still display empty counts
            draw_boxes(im0, [], self.model.names, [], None, frame_counts={})

        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    init_tracker()
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()
