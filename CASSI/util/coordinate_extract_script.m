clear;

a = imread('flower.JPG');

figure; imshow(a)
num_coords = 2;

rectobj = zeros(num_coords,4);
for i = 1:num_coords
[~, rectobj(i,:)] = imcrop;
end

rectarr = round(rectobj);
rectarr(:,[1,2]) = rectarr(:,[2,1]);
rectarr(:,[2,3]) = rectarr(:,[3,2]);
rectarr(:,[2,4]) = rectarr(:,[1,3]) + rectarr(:,[4,2]);