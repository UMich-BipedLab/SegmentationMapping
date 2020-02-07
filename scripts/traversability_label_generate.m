clear;
clc;

root = "/home/ganlu/Datasets/Traversability/";
rgb_folder = root + "rgb/";
segmentation_folder = root + "seg/";
projection_folder = root + "proj/";
final_folder = root + "labels/train/";
final_rgb_folder = root + "images/train/";

images = dir(fullfile(rgb_folder, '*.png'));
for i = 1: size(images)
    image_name = images(i).name;
    rgb_img = imread(rgb_folder + image_name);
    try
        segmentation_img = imread(segmentation_folder + image_name);
        projection_img = imread(projection_folder + image_name);
        final_img = uint8(and(segmentation_img, projection_img));
        %imagesc(final_img)
        imwrite(final_img, final_folder + image_name);
        imwrite(rgb_img, final_rgb_folder + image_name);
    catch
        continue
    end
end
