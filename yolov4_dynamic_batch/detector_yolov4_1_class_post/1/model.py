import numpy as np
import sys
import json

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get YOLO_01 configuration
        input0_config = pb_utils.get_input_config_by_name(model_config, "YOLO_01")
        # Get YOLO_02 configuration
        input1_config = pb_utils.get_input_config_by_name(model_config, "YOLO_02")
        # Get YOLO_02 configuration
        input2_config = pb_utils.get_input_config_by_name(model_config, "YOLO_03")

        # Get INPUT3 configuration
        input3_config = pb_utils.get_input_config_by_name(model_config, "ORIG_SHAPE")
        # Get INPUT4 configuration
        input4_config = pb_utils.get_input_config_by_name(model_config, "POST_SHAPE")

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT0")
        # Get OUTPUT1 configuration
        output1_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT1")
        # Get OUTPUT2 configuration
        output2_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT2")
        # Get OUTPUT3 configuration
        output3_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT3")

        # Convert Triton types to numpy types
        self.input0_dtype = pb_utils.triton_string_to_numpy(input0_config['data_type'])
        self.input1_dtype = pb_utils.triton_string_to_numpy(input1_config['data_type'])
        self.input2_dtype = pb_utils.triton_string_to_numpy(input2_config['data_type'])
        self.input3_dtype = pb_utils.triton_string_to_numpy(input3_config['data_type'])
        self.input4_dtype = pb_utils.triton_string_to_numpy(input4_config['data_type'])

        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config['data_type'])
        self.output1_dtype = pb_utils.triton_string_to_numpy(output1_config['data_type'])
        self.output2_dtype = pb_utils.triton_string_to_numpy(output2_config['data_type'])
        self.output3_dtype = pb_utils.triton_string_to_numpy(output3_config['data_type'])

        # Instantiate the postprocess model
        self.model = 'detector_yolov4_1_class_post'

    def execute(self, requests):
        input0_dtype = self.input0_dtype
        input1_dtype = self.input1_dtype
        input2_dtype = self.input2_dtype
        input3_dtype = self.input3_dtype
        input4_dtype = self.input4_dtype

        output0_dtype = self.output0_dtype
        output1_dtype = self.output1_dtype
        output2_dtype = self.output2_dtype
        output3_dtype = self.output3_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0 (Classes)
            in_0 = pb_utils.get_input_tensor_by_name(request, "YOLO_01")
            # Get INPUT1 (Boxes)
            in_1 = pb_utils.get_input_tensor_by_name(request, "YOLO_02")
            # Get INPUT2 (Scores)
            in_2 = pb_utils.get_input_tensor_by_name(request, "YOLO_03")
            # Get INPUT3 (Original Shape)
            in_3 = pb_utils.get_input_tensor_by_name(request, "ORIG_SHAPE")
            # Get INPUT4 (Postprocess Shape)
            in_4 = pb_utils.get_input_tensor_by_name(request, "POST_SHAPE")

            orig_size = in_3.as_numpy().astype(input3_dtype)[0][::-1] # like (h, w)
            post_size = in_4.as_numpy().astype(input4_dtype)[0][::-1] # like (h, w)
            # try: # ValueError
            model_outputs = [in_0.as_numpy().astype(input0_dtype),
                             in_1.as_numpy().astype(input1_dtype),
                             in_2.as_numpy().astype(input2_dtype)]
            # raise ValueError('model outputs:', in_0.as_numpy().astype(input0_dtype).shape)
            print(f"{orig_size} | {post_size}")
            boxes, scores, labels, indexes = Processing._postprocess_yolo_batch(model_outputs, post_size, orig_size,
                                                                conf_th=0.5,  nms_threshold=0.5)
            # raise ValueError(f"{boxes} | {scores} | {labels} | {orig_size} | {post_size} | {indexes}")
            print(f"{boxes} | {scores} | {labels} | {orig_size} | {post_size} | {indexes}")
            # Create output tensors. You need pb_utils.Tensor objects to create pb_utils.InferenceResponse.
            out_tensor_0 = pb_utils.Tensor("OUTPUT0", boxes.astype(output0_dtype))  # out_0.astype(output0_dtype))
            out_tensor_1 = pb_utils.Tensor("OUTPUT1", scores.astype(output1_dtype))  # out_1.astype(output1_dtype))
            out_tensor_2 = pb_utils.Tensor("OUTPUT2", labels.astype(output2_dtype))
            out_tensor_3 = pb_utils.Tensor("OUTPUT3", indexes.astype(output3_dtype))
            # Create InferenceResponse. You can set an error here in case there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference response:
            #
            # pb_utils.InferenceResponse(output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0, out_tensor_1, out_tensor_2, out_tensor_3])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')


class Processing:
    """Users class for making processing & postprocessing.
    Using inside "TritonPythonModel".
    """
    @staticmethod
    def nms_boxes(detections, nms_threshold):
        """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding
        boxes with their confidence scores and return an array with the
        indexes of the bounding boxes we want to keep.

        # Args
            detections: Nx7 numpy arrays of
                        [[x, y, w, h, box_confidence, class_id, class_prob],
                         ......]
        """
        x_coord = detections[:, 0]
        y_coord = detections[:, 1]
        width = detections[:, 2]
        height = detections[:, 3]
        box_confidences = detections[:, 4] * detections[:, 6]

        areas = width * height
        ordered = box_confidences.argsort()[::-1]

        keep = list()
        while ordered.size > 0:
            # Index of the current element:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
            yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
            xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
            yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1
            union = (areas[i] + areas[ordered[1:]] - intersection)
            iou = intersection / union
            indexes = np.where(iou <= nms_threshold)[0]
            ordered = ordered[indexes + 1]

        keep = np.array(keep)
        return keep

    @staticmethod
    def postprocess_yolo(trt_outputs, input_shape, output_shape, conf_th, nms_threshold, letter_box=False):
        """Postprocess TensorRT outputs.

        # Args
            trt_outputs: a list of 2 or 3 tensors, where each tensor
                        contains a multiple of 7 float32 numbers in
                        the order of [x, y, w, h, box_confidence, class_id, class_prob]
            conf_th: confidence threshold
            letter_box: boolean, referring to _preprocess_yolo()

        # Returns
            boxes, scores, classes (after NMS)
        """
        img_w, img_h = output_shape
        # filter low-conf detections and concatenate results of all yolo layers
        detections = []
        for o in trt_outputs:
            dets = o.reshape((-1, 7))
            dets = dets[dets[:, 4] * dets[:, 6] >= conf_th]
            detections.append(dets)
        detections = np.concatenate(detections, axis=0)

        if len(detections) == 0:
            boxes = np.zeros((0, 4), dtype=np.int)
            scores = np.zeros((0,), dtype=np.float32)
            classes = np.zeros((0,), dtype=np.float32)
        else:
            box_scores = detections[:, 4] * detections[:, 6]

            # scale x, y, w, h from [0, 1] to pixel values
            old_h, old_w = img_h, img_w
            offset_h, offset_w = 0, 0
            if letter_box:
                if (img_w / input_shape[1]) >= (img_h / input_shape[0]):
                    old_h = int(input_shape[0] * img_w / input_shape[1])
                    offset_h = (old_h - img_h) // 2
                else:
                    old_w = int(input_shape[1] * img_h / input_shape[0])
                    offset_w = (old_w - img_w) // 2
            detections[:, 0:4] *= np.array(
                [old_w, old_h, old_w, old_h], dtype=np.float32)

            # NMS
            nms_detections = np.zeros((0, 7), dtype=detections.dtype)
            for class_id in set(detections[:, 5]):
                idxs = np.where(detections[:, 5] == class_id)
                cls_detections = detections[idxs]
                keep = Processing.nms_boxes(cls_detections, nms_threshold)
                nms_detections = np.concatenate(
                    [nms_detections, cls_detections[keep]], axis=0)

            xx = nms_detections[:, 0].reshape(-1, 1)
            yy = nms_detections[:, 1].reshape(-1, 1)
            if letter_box:
                xx = xx - offset_w
                yy = yy - offset_h
            ww = nms_detections[:, 2].reshape(-1, 1)
            hh = nms_detections[:, 3].reshape(-1, 1)
            boxes = np.concatenate([xx, yy, xx + ww, yy + hh], axis=1) + 0.5
            boxes = boxes.astype(np.int)
            scores = nms_detections[:, 4] * nms_detections[:, 6]
            classes = nms_detections[:, 5]
        return boxes, scores, classes

    @staticmethod
    def _postprocess_yolo_batch(trt_outputs, input_shape, output_shape, conf_th, nms_threshold, letter_box=False):
        bn, *_ = trt_outputs[0].shape
    
        all_boxes, all_scores, all_classes = [], [], []
        indexes = []
        for bi in range(bn):
            boxes, scores, classes = Processing.postprocess_yolo([out[bi] for out in trt_outputs], input_shape, output_shape, conf_th, nms_threshold, letter_box=False)
            all_boxes.append(boxes)
            all_scores.extend(scores.reshape(-1))
            all_classes.extend(classes.reshape(-1))
            indexes.extend(len(boxes) * [bi])
        return np.vstack(all_boxes), np.asarray(all_scores), np.asarray(all_classes), np.asarray(indexes)