import numpy as np
from dataclasses import dataclass
from ultralytics import YOLO
from app.utils.logger import logger


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    def to_tuple(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class YOLODetector:
    """YOLOv8 person detector. CPU-optimised for development."""

    PERSON_CLASS_ID = 0

    def __init__(self, weights_path: str = "weights/yolov8n.pt",
                 confidence: float = 0.5, device: str = "cpu"):
        self.confidence = confidence
        self.device = device
        logger.info(f"Loading YOLO model from {weights_path} on {device}...")
        # ultralytics auto-downloads yolov8n.pt if not found
        self.model = YOLO(weights_path)
        self.model.to(device)
        logger.info("YOLO model loaded")

    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        """Run inference, return only person detections."""
        results = self.model(
            frame,
            conf=self.confidence,
            classes=[self.PERSON_CLASS_ID],
            verbose=False,
            device=self.device,
        )

        boxes: list[BoundingBox] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                boxes.append(BoundingBox(x1, y1, x2, y2, conf))
        return boxes

    def draw_boxes(self, frame: np.ndarray, boxes: list[BoundingBox],
                   label_prefix: str = "Person") -> np.ndarray:
        """Draw bounding boxes on frame for debugging."""
        import cv2
        annotated = frame.copy()
        for i, box in enumerate(boxes):
            cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
            label = f"{label_prefix} {i + 1} ({box.confidence:.0%})"
            cv2.putText(annotated, label, (box.x1, box.y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return annotated
