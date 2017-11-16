% f is the index of the feature
% i is the index of the iimage
function res = computeFeature(i,f,featuretbl,iimages)
    iimage = reshape(iimages(i,:),[64,64])';
    ul_w_r = featuretbl(f,1);
    ul_w_c = featuretbl(f,2);
    ul_b_r = featuretbl(f,3);
    ul_b_c = featuretbl(f,4);
    lr_w_r = featuretbl(f,5);
    lr_w_c = featuretbl(f,6);
    lr_b_r = featuretbl(f,7);
    lr_b_c = featuretbl(f,8);
    ur_w_r = ul_w_r;
    ur_w_c = lr_w_c;
    ll_w_r = lr_w_r;
    ll_w_c = ul_w_c;
    white = iimage(lr_w_r,lr_w_c) + iimage(ul_w_r,ul_w_c) - iimage(ur_w_r,ur_w_c) - iimage(ll_w_r,ll_w_c);
    ur_b_r = ul_b_r;
    ur_b_c = lr_b_c;
    ll_b_r = lr_b_r;
    ll_b_c = ul_b_c;
    black = iimage(lr_b_r,lr_b_c) + iimage(ul_b_r,ul_b_c) - iimage(ur_b_r,ur_b_c) - iimage(ll_b_r,ll_b_c);
    res = white - black;
end

