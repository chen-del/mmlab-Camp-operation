python tools/train.py ./Balloon_Drink/configs/rtmdet_tiny_balloon.py
# python tools/train.py ./Balloon_Drink/configs/rtmdet_tiny_drink.py


# python /home/hanhan/cc_wrod/develop/mmlab/mmdetection/demo/image_demo.py \
#         ./datasets/Balloon/images.jpg \
#         ./Balloon_Drink/configs/rtmdet_tiny_balloon.py \
#         --weights  ./work_dirs/rtmdet_tiny_balloon/best_coco_bbox_mAP_epoch_40.pth \
#         --out-dir outputs/ \
#         --device cuda:0 \
#         --pred-score-thr 0.5


# python /home/hanhan/cc_wrod/develop/mmlab/mmdetection/tools/test.py ./Balloon_Drink/configs/rtmdet_tiny_balloon.py  \
#                       ./work_dirs/rtmdet_tiny_balloon/best_coco_bbox_mAP_epoch_40.pth