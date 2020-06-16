clear all;
close all;
clc;

% Number of pixel in each dimension (A squared image with 256*256 pixels)
IM_Size =32;
% wavelength and wavenumber of the illumination light
lambda=532*10^-9;
k0=2*pi/lambda;

% Object information
% xy coordinates
x=1:IM_Size;
y=1:IM_Size;
% xy resolutions
dx=5*10^(-6);
dy=5*10^(-6);

iu=sqrt(-1);

% load the video from CUP imaging
% 80 frames of jelly fish swimming (each frame has 256*256 pixels after imresize)
load('russianvids.mat')
vids = 255.*double(sample);
vids = permute(vids,[1,3,4,2]);

% number of pixels in vertical direction in the CUP image (the image after streaking)
IM_Size1 = IM_Size + size(vids,4)-1;
% add a background of 50
I = vids;

% load the DMD mask
%load('Mask1.mat');
load('binary_ball_mask.mat');
Mask1 = double(Mask1);



%% manually building the streaking image
for vid =1:size(vids,1)
    sample = reshape(vids(vid,:,:,:),[32,32,30]);
 
    
    for i = 1:size(sample,3)
    I_dmd(:,:,i) = reshape(I(vid,:,:,i),[32,32]);
    I_dmd_prime(:,:,i) = reshape(I(vid,:,:,i),[32,32])';
    end

    % plot of one frame of the jelly fish
    %figure;imagesc(abs(I_dmd(:,:,1))); axis square;axis off; colormap hot;

    I_dmd_full = zeros(IM_Size1,IM_Size,size(sample,3));
    Mask1_full = zeros(IM_Size1,IM_Size,size(sample,3));

    for i = 1:size(sample,3)
        I_dmd_full((i-1)+1:(i-1)+IM_Size,:,i) = I_dmd(:,:,i);
    end
    for i = 1:size(sample,3)
        Mask1_full((i-1)+1:(i-1)+IM_Size,:,i) = Mask1;
    end

    I_dmd_shear = I_dmd_full.*Mask1_full;

    I_CCD = sum(I_dmd_shear,3);
    I_CCD_vector = I_CCD(:);
    % the streaking image
    %figure;imagesc(I_CCD);axis equal;axis off;colormap hot;
    streaked_ims(vid,:) = I_CCD_vector;
end
%% Build the forward model matrix
for i = 1:size(sample,3)
    Mask_temp = Mask1';
    Mask2_full_vector(:,i) = Mask_temp(:);
end

count = 1;
for i = 1:size(sample,3)
    for j = 1:IM_Size*IM_Size
        y_coordinate(count) = j + (i-1)*IM_Size;
        x_coordinate(count) = (i-1)*IM_Size*IM_Size + j;
        A_coordinate(count) = Mask2_full_vector(j,i);
        count = count + 1;
    end
end

% forward model matrix
S_forward=sparse(y_coordinate,x_coordinate,A_coordinate,IM_Size*IM_Size1,IM_Size*IM_Size*size(sample,3));

% the streaking image from the forward model matrix
y_verify = reshape(S_forward*I_dmd_prime(:),IM_Size,IM_Size1);
y_verify = y_verify';
%%
% to verify if it is consistent with the manually created streaking image
% if the plot is all 0, it means no error
figure;imagesc(y_verify-I_CCD); axis equal;axis off;colormap hot;
I_CCD = I_CCD';
I_CCD_vector = I_CCD(:);% vectorize the streaking image

%% calculate the eigenvalues of the forward model matrix
para = 4.7; % tune this parameter to make the largest eigenvalue of A to be < 0
S = S_forward./para;
A = S'*S;
eigs(A)
eg_all = eigs(A); % save it as 'eg_all.mat' and load it later
save('eg_all.mat','eg_all');
%%
para = 4.7;
S_forward = S_forward./para;
y_meas = streaked_ims'./para;

%%
streaked_ims = streaked_ims';
for i =1:size(streaked_ims,2)
    estimates(:,i) = lsqr(S_forward,streaked_ims(:,i),[],50);
end