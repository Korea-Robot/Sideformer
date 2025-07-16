# python inference_visualize.py --model_path ckpts/ddp_seg_model_epoch_70.pth --device cuda:4 --num_samples 20 --save_dir inference_results_best_ddp
# python inference_visualize.py --model_path ckpts/seg_model_epoch_70.pth --device cuda:4 --num_samples 20 --save_dir inference_results_best
python inference_visualize.py --model_path ckpts/best_seg_model.pth --device cuda:4 --num_samples 20 --save_dir inference_results_best

python inference_visualize.py --model_path ckpts/seg_model_epoch_70.pth --device cuda:4 --num_samples 20 --save_dir inference_results_70
# python inference_visualize.py --model_path ckpts/best_seg_model.pth --device cuda:4 --num_samples 20 --save_dir inference_results_best
