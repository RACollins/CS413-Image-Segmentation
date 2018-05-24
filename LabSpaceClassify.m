%% Clear all variables and images
% 
% 
clc;    % Clear the command window.
close all;  % Close all figures.
clear;  % Clear all existing variables.
workspace;  % Make sure the workspace panel is showing.

%% Prompts, Opening file, and Adding Gaussian
% This Section opens the image file and aplies Gaussian noise if indicated
% by the user.
%
% Promts are kept at the top of the code for easy reference. 

filePrompt = 'Please enter file name with extension: ';
clusterPrompt = 'Please enter number of colours: ';
gaussianPrompt = 'Add Gaussian noise? (y/n): ';
snrPrompt = 'Please enter signal-to-noise ratio: ';
cropPrompt = 'Neatly organise objects into a seperate images? (y/n): ';

%%
%
% Read image file
fileName = input(filePrompt, 's');
I = imread(fileName);

%%
%
% Add Gaussian noise
noiseCondition = input(gaussianPrompt, 's');
if strcmp(noiseCondition, 'y')
    snr = input(snrPrompt);
    I_double = im2double(I); %Convert to double to calculate varI
    varI = std2(I_double)^2;
    sigma_noise = sqrt(varI/snr);
    I = imnoise(I, 'gaussian', 0, sigma_noise^2); %Add noise to I
elseif strcmp(noiseCondition, 'n')
    disp('No noise added');
else
    disp('Invalid input, no noise added');
end

%% Cluster by Colour
% This section deals with the identification and clustering of colours
%
% Convert image to 'Lba' colour space
cform = makecform('srgb2lab');
lab_I = applycform(I, cform);

%%
%
% Classify the colours in 'ab' space using k-means clustering
ab = double(lab_I(:, :, 2:3));
nRows = size(ab, 1);
nCols = size(ab, 2);
ab = reshape(ab, nRows*nCols, 2);
nColours = input(clusterPrompt);
% Repeat the clustering 3 times to avoid local minima
cluster_idx = kmeans(ab, nColours, 'distance', 'sqEuclidean', ...
                                      'Replicates', 3);
%%
%
% Label every pixel using the results from k-means                                  
pixel_labels = reshape(cluster_idx, nRows, nCols);
imshow(pixel_labels, []), title('image labeled by cluster index');

%% 
%
% Create images that segment by colour.
segmented_images = cell(1, 3);
rgb_label = repmat(pixel_labels, [1 1 3]);

for k = 1:nColours
    %%
    %
    % Blackout all pixel except those in each cluster
    colour = I;
    colour(rgb_label ~= k) = 0;
    segmented_images{k} = colour;
    %%
    %
    % Create binary mask and remove all objects smaller than x pixels
    segmented_images_grey = rgb2gray(segmented_images{k});
    segmented_images_BW = imbinarize(segmented_images_grey,'adaptive', ...
        'ForegroundPolarity', 'dark', 'Sensitivity', 0.4);
    segmented_images_BW = bwareaopen(segmented_images_BW, 1400);
    %%
    %
    % Crop and organise objects by colour
    cropCondition = input(cropPrompt, 's');
    if strcmp(cropCondition, 'y')
        labeledI = logical(segmented_images_BW);
        blobMeasurements = regionprops(labeledI, 'BoundingBox');
        nBlobs = size(blobMeasurements, 1);	
        for blob = 1:nBlobs  % Loop through all blobs.
            % Find the bounding box of each blob.
            thisBlobsBoundingBox = blobMeasurements(blob).BoundingBox;  % List of pixels in current blob.
            % Extract this sweet into it's own image.
            subImage = imcrop(I, thisBlobsBoundingBox);
            % Display the image.
            subplot(ceil(sqrt(nBlobs)), ceil(sqrt(nBlobs)), blob);
            imshow(subImage);
        end
    elseif strcmp(cropCondition, 'n')
        disp('Objects not organised');
        % Count objects
        cc = bwconncomp(segmented_images_BW, 4);
        % Print cleaned segmented colours
        t = sprintf('%s object(s) in cluster number %s', ...
            num2str(cc.NumObjects), num2str(k));
        segmented_images_COLOUR = segmented_images_BW.*im2double(I);
        figure; imshow(segmented_images_COLOUR), title(t);
    else
        disp('Invalid input, no noise added');
        % Count objects
        cc = bwconncomp(segmented_images_BW, 4);
        % Print cleaned segmented colours
        t = sprintf('%s object(s) in cluster number %s', ...
            num2str(cc.NumObjects), num2str(k));
        segmented_images_COLOUR = segmented_images_BW.*im2double(I);
        figure; imshow(segmented_images_COLOUR), title(t);
    end
end
