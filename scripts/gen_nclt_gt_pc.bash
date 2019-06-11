#python get_projected_gt.py --input_pointcloud ~/rvm_nclt.txt --rgb_img ~/Documents/nclt_0429_rvm/1335704515132779_rgb.tiff  --gt_img ~/Documents/nclt_0429_rvm/1335704515132779_bw.png  --cam2lidar_projection ~/perl_code/workspace/src/segmentation_projection/config/nclt_cams/cam2lidar_4.npy  --cam_intrinsic ~/perl_code/workspace/src/segmentation_projection/config/nclt_cams/nclt_intrinsic_4.npy  --distortion_map ~/perl_code/workspace/src/segmentation_projection/config/nclt_cams/U2D_Cam4_1616X1232.txt  --output_pointcloud out.txt

nclt_eval_root=$1 # root folder containing configs, 
seg_lidar_pts=$2
output=$3

python get_projected_gt.py  --segmented_lidar_folder $2 --gt_folder $1/gt_may23/ --config_folder $1/config/ --output_folder $3/ --rgb_img_folder $1/rgb_may23
