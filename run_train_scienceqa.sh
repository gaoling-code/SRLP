# TRAIN - rationale generation
# CUDA_VISIBLE_DEVICES=0,1 python main.py --model ./models/unifiedqa-t5-base --user_msg rationale --img_type detr --bs 8 --eval_bs 4 --eval_acc 10 --output_len 512 --final_eval --prompt_format QCM-LE --epoch 50 --vot_num 5 --alpha 0.5 --output_dir ./results/code5_train

# Train - answer inference
CUDA_VISIBLE_DEVICES=0 python main.py --model ./models/unifiedqa-t5-base --user_msg answer --img_type detr --bs 16 --eval_bs 24 --eval_acc 10 --output_len 64 --final_eval --prompt_format QCMG-A --epoch 50 --vot_num 5 --alpha 0.5 --eval_le ./results/code5_train/rationale/predictions_ans_eval.json --test_le ./results/code5_train/rationale/predictions_ans_test.json --output_dir ./results/code5_train
