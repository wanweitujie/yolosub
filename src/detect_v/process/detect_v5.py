from yolov5.utils.general import (LOGGER, check_img_size,
                        non_max_suppression, scale_coords)
from yolov5.utils.torch_utils import time_sync
from yolov5.utils.augmentations import (letterbox)    

import torch
import cv2
import numpy as np


def detect_v5_model(model: any,img:cv2.typing.MatLike, config:dict[str,any],decimals : int = 5):
    # InferenceConfig 字段名(conf/iou)与 yolov5 detect() 参数名(conf_thres/iou_thres)不一致,
    # 在此做一次映射; 其余同名参数(imgsz/max_det/classes/agnostic_nms/augment)直传。
    # half/retina_masks/embed 等 yolov5 detect() 不支持的参数自动丢弃, 避免触发 unexpected keyword argument。
    v5_args: dict[str, any] = {}
    if config:
        if "conf" in config: v5_args["conf_thres"] = config["conf"]
        if "iou" in config: v5_args["iou_thres"] = config["iou"]
        for k in ("imgsz", "max_det", "classes", "agnostic_nms", "augment"):
            if k in config: v5_args[k] = config[k]
    return detect(model,img,decimals,**v5_args)

def yolov5_summary(det, names, decimals=5, img_shape = None, normalize=False):
    """
    模拟 YOLOv8 的 results[0].summary()
    
    Args:
        det: Tensor (N, 6) → [xyxy, conf, cls]
        names: dict 或 list，类别名
        img_shape: (H, W) 原图尺寸
        normalize: False → 绝对坐标, True → 归一化 [0,1]
        decimals: 小数位
    
    Returns:
        list: 检测结果列表，每个元素包含检测对象信息
    """

    if det is None or len(det) == 0:
        return []

    det = det.cpu()

    # 提取字段
    xyxy = det[:, :4]
    conf = det[:, 4]
    cls = det[:, 5].long()

    # # 归一化坐标 (optional)
    # if normalize:
    #     h, w = img_shape
    #     xyxyn = xyxy / torch.tensor([w, h, w, h])
    # else:
    #     xyxyn = xyxy.clone()

    # 转为 list + 保留小数
    xyxy_list = xyxy.tolist()
    # xyxyn_list = xyxyn.tolist()
    conf_list = conf.tolist()
    cls_list = cls.tolist()
    names_list = [names[int(c)] for c in cls]

    # 构建新的返回结构
    results = []
    for i in range(len(det)):
        result = {
            "name": names_list[i],
            "class": cls_list[i],
            "confidence":round(conf_list[i], decimals),
            "box": {
                "x1": round(xyxy_list[i][0], decimals),
                "y1": round(xyxy_list[i][1], decimals),
                "x2":round( xyxy_list[i][2], decimals),
                "y2":round( xyxy_list[i][3], decimals)
            }
        }
        results.append(result)

    return results

def detect(model:any, img0:cv2.typing.MatLike, decimals:int,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        ):

    # global_logger.info(f"model.names type:{ type(model.names)}")
    # global_logger.info(f"model.names:{model.names}")
    # global_logger.info(f"model.names length:{len(model.names) if hasattr(model.names, '__len__') else 'N/A'}")
    

    # Load model
    device = model.device
    stride, names, pt = model.stride, model.names, model.pt
    if isinstance(names, dict):
        names = list(names.values())
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    img_shape=img0.shape
    img_hw = img_shape[:2]

    img = letterbox(img0, imgsz, stride=stride, auto=pt)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    s = f'image: '

    # # Dataloader    
    # dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, dt = 0, [0.0, 0.0, 0.0]
    visualize = False

    #for _, im, im0s, _, _ in dataset:
    t1 = time_sync()
    img = torch.from_numpy(img).to(device)
    img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1

    # Inference
    pred = model(img, augment=augment, visualize=visualize)
    t3 = time_sync()
    dt[1] += t3 - t2

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    dt[2] += time_sync() - t3

    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
    result = None
    # Process predictions
    for i, det in enumerate(pred):  # per image
        seen += 1
        # im0 = im0s.copy()

        s += '%gx%g ' % img.shape[2:]  # print string

        if len(det):
            # Rescale boxes from img_size to im0 size
            
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_shape # im0.shape
                                        ).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            result = yolov5_summary(det,names,decimals=decimals,img_shape=img_hw,normalize=False)
            # # Write results
            # for *xyxy, conf, cls in reversed(det):
            #     pass

             

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    # 返回 (框列表, speed_ms) 二元组:第三段为 NMS(与 ultralytics 的 postprocess 语义对齐),
    # 由 gpu_manager.detect 拆包存入 model_info.last_speed_ms,detect 对外响应结构不变。
    # 与 ultralytics results[0].summary() 行为对齐:无检测返回 [] 而非 None
    return (result or []), {"preprocess": t[0], "inference": t[1], "postprocess": t[2]}