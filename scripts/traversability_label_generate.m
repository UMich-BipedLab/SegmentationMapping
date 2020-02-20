clear;
clc;

root = "/home/ganlu/Datasets/Traversability/";
rgb_folder = root + "rgb/";
segmentation_folder = root + "seg/";
projection_folder = root + "proj/";
label_train_folder = root + "labels/train/";
image_train_folder = root + "images/train/";
label_val_folder = root + "labels/val/";
image_val_folder = root + "images/val/";

images = dir(fullfile(rgb_folder, '*.png'));

%% Generate training data
for i = 1: size(images)
    image_name = images(i).name;
    %image_name = '1569971103023503552.png';
    rgb_img = imread(rgb_folder + image_name);
    try
        segmentation_img = imread(segmentation_folder + image_name);
        projection_img = imread(projection_folder + image_name);
        final_img = uint8(and(segmentation_img, projection_img));
        imagesc(final_img)
        imwrite(final_img, label_train_folder + image_name);
        imwrite(rgb_img, image_train_folder + image_name);
    catch
        continue
    end
end

%% Generate validation data
percentage = 0.05;
valid_size = round(percentage * size(images, 1));
valid_images = round(rand(valid_size, 1) * valid_size);
for i = 1 : size(valid_images)
    image_name = images(valid_images(i)).name;
    rgb_img = imread(rgb_folder + image_name);
    try
        segmentation_img = imread(segmentation_folder + image_name);
        projection_img = imread(projection_folder + image_name);
        final_img = uint8(and(segmentation_img, projection_img));
        %imagesc(final_img)
        imwrite(final_img, label_val_folder + image_name);
        imwrite(rgb_img, image_val_folder + image_name);
    catch
        continue
    end
end

%% Generate visuals
colored_img = rgb_img;
for i = 1 : size(final_img, 1)
    for j = 1 : size(final_img, 2)
        if (projection_img(i,j) == 0)
            %colored_img(i, j, 1) = 128;
            %colored_img(i, j, 2) = 62;
            %colored_img(i, j, 3) = 128;
            colored_img(i, j, 1) = 255;
            colored_img(i, j, 2) = 0;
            colored_img(i, j, 3) = 0;
        else
            %colored_img(i, j, 1) = 235;
            %colored_img(i, j, 2) = 51;
            %colored_img(i, j, 3) = 232;
            colored_img(i, j, 1) = 0;
            colored_img(i, j, 2) = 0;
            colored_img(i, j, 3) = 255;
        end

    end    
end

combined_img = colored_img * 0.5 + rgb_img * 0.5;
imshow(combined_img)