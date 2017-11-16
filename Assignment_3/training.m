%For the iimages, 
%all with odd indices are nonfaces
%all with even indices are faces
N = 2000;%total number of images in the training set
T = 20; %T being the maximum number of boosting
cascades = 5;
iimages = dlmread('iimages.txt');
featuretbl = dlmread('featuretbl.txt');
labels = zeros([1,4000]);
parfor i = 1:4000
    if mod(i,2) == 1
        labels(i) = -1;
    else
        labels(i) = 1;
    end
end

res = cascade(featuretbl, iimages, labels, cascades, N, T);
save('res.mat','res');