out_ssim = []
out_ind = []
out_mse = []
for out=1:9
    fnum = out*out*8
    ssim_outer = []
    mse_outer = []
    arr_ind = 1
    for outer=0:1:30
        clearvars -EXCEPT outer ssim_outer mse_outer arr_ind out out_ind out_ssim out_mse fnum;
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
        file_load = "ball_training_5_4/test_ball_video_"+fnum+".mat"
        load(file_load)
        sample = 255.*double(sample);
        sample = permute(sample,[2,3,1]);
        for i = 1:size(sample,3)
            I_temp = sample(:,:,i);
            sample_temp(:,:,i) = imresize(I_temp,[32 32]);
        end
        sample = sample_temp;

        % number of pixels in vertical direction in the CUP image (the image after streaking)
        IM_Size1 = IM_Size + size(sample,3)-1;
        % add a background of 50
        %sample = sample + 50;
        I = sample;

        % load the DMD mask
        %load('Mask1.mat');
        load('ball_training_5_4/binary_ball_mask.mat');
        Mask1 = double(Mask1);

        %% manually building the streaking image
        for i = 1:size(sample,3)
        I_dmd(:,:,i) = I(:,:,i);
        I_dmd_prime(:,:,i) = I(:,:,i)';
        end

        % plot of one frame of the jelly fish
        figure;imagesc(abs(I_dmd(:,:,1))); axis square;axis off; colormap hot;

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
        figure;imagesc(I_CCD);axis equal;axis off;colormap hot;


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
        y_meas = I_CCD_vector./para;
        initial_guess = S_forward'*y_meas; % initial guess of the result from back-projection
        initial_guess_2D = reshape(initial_guess,IM_Size,IM_Size,size(sample,3));
        initial_guess_2D = permute(initial_guess_2D,[2 1 3]);

    %     v0 = VideoWriter('Intensity_backprojection.avi')
    %     v0.FrameRate = 3;
    %     open(v0);
    %     for i = 1:size(sample,3)+1
    %         if i==1
    %             figure(1);imagesc(zeros(32,32));axis off;axis equal;colormap bone;
    %         else
    %         figure(1);imagesc(abs(initial_guess_2D(:,:,i-1))); axis off;axis equal;colormap gray;
    %         end
    %         frame = getframe(gcf);
    %        writeVideo(v0,frame);
    %     end
    %     close(v0);

        tau = outer/(para^2)   % tune tau until having a reasonable reconstruction result
        load('eg_all.mat');
        %lambda1 = eg_all(end);  % the smallest eigenvalue of A
        lambda1 = 1e-4;
        tolA = 1e-12; % tolerance in the stop criterion
        tv_iters1=2;

        % Define the regularizer
        Psi1 = @(x,th)  denoise_func_new(x,th,IM_Size,IM_Size,size(sample,3),tv_iters1);
        Phi = @(x) MyTVphi_new(x,IM_Size,IM_Size,size(sample,3));

        % Run TwIST 

        [reconstruction_Image,x_debias_twist,obj_twist,...
            times_twist,debias_start_twist,mse]= ...
                 TwIST(y_meas,S_forward,tau, ...
                 'Lambda', lambda1, ...
                 'Debias',0,...  
                 'Monotone',1,...
                 'Sparse', 1,...
                 'Psi',Psi1,...
                 'Phi',Phi,...
                 'Initialization',initial_guess,...   
                 'StopCriterion',1,...
                 'ToleranceA',tolA,...
                 'Verbose', 0);

        reconstruction_Image = reshape(reconstruction_Image,IM_Size,IM_Size,size(sample,3));


        for i = 1:size(sample,3)
            reconstruction_Image_final(:,:,i) = reconstruction_Image(:,:,i)';
        end

    %     v = VideoWriter('Intensity_groundtruth_93.avi')
    %     v.FrameRate = 3;
    %     open(v);
    %     for i = 1:size(sample,3)+1
    %         if i==1
    %             figure(1);imagesc(zeros(32,32));axis off;axis square;colormap bone;
    %         else
    %         figure(1);imagesc(abs(I_dmd(:,:,i-1))); axis off;axis square;colormap gray;
    %         end
    %         frame = getframe(gcf);
    %        writeVideo(v,frame);
    %     end
    %     close(v);
    % 
    %     v2 = VideoWriter('Intensity_TwIST_93.avi')
    %     v2.FrameRate = 3;
    %     open(v2);
    %     for i = 1:size(sample,3)+1
    %         if i==1
    %             figure(2);imagesc(zeros(32,32));axis off;axis square;colormap bone;
    %         else
    %         figure(2);imagesc(abs(reconstruction_Image_final(:,:,i-1))); axis off;axis square;colormap gray;
    %         end
    %         frame = getframe(gcf);
    %        writeVideo(v2,frame);
    %     end
    %     close(v2);
       ssim_outer(arr_ind) = ssim(I_dmd,reconstruction_Image_final)
       mse_outer(arr_ind) = immse(I_dmd,reconstruction_Image_final)
       arr_ind = arr_ind + 1

    end
   [mx,zeta] = max(ssim_outer);
   out_ssim(out) = mx
   out_mse(out) = mse_outer(zeta)
   out_ind(out) = zeta
   
end

