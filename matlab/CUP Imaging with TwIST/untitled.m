estimates = zeros(233,32,32,30);
for i=1:233
    file = "lsqr_recons2/"+i+".mat"
    load(file)
    estimates(i,:,:,:) = ig2d(:,:,:);
end    
estimates = single(estimates);
save('lsqr_jelly_estimates.mat','estimates')
   
    