%featuretbl, iimages and labels are original data
%cas is the number of cascades
%N is the numner of images
%T is the number of weak learners
function res = cascade(featuretbl, iimages, labels, cas, N, T)
    cas_number = N;
    cas_labels = labels(1:N);
    cas_iimages = iimages(1:N,:);
    res = cell(cas,1);
    for i = 1:cas
        fprintf('cascade: %d\n',i);
        predictionArray = zeros([1,cas_number]);
        [alphas, weakLearners] = adaBoost(T,cas_number, featuretbl, cas_iimages, cas_labels);
        curr_T = length(alphas);
        res{i} = [alphas, reshape(weakLearners,[1,3*curr_T])];
        thetas = zeros([1,cas_number]);
        k = 1;
        for j = 1:cas_number
            if labels(j) == -1
                continue
            end
            thetas(k) = strongLearner_helper(alphas, weakLearners,j,featuretbl,iimages);
            k = k+1;
        end
        thetas(k:cas_number) = [];
        theta = min(thetas);
        for j = 1:cas_number
            predictionArray(j) = strongLearner(alphas, weakLearners,j,featuretbl,cas_iimages,theta);
        end
        deleteArray = zeros([1,cas_number]);
        k = 1;
        for j = 1:cas_number
            if predictionArray(j) == -1 && cas_labels(j) == -1
                deleteArray(k) = j;
                k = k+1;
            end
        end
        deleteArray(k:cas_number) = [];
        cas_labels(:,deleteArray) = [];
        cas_iimages(deleteArray,:) = [];
        cas_number = cas_number - length(deleteArray);
        fprintf('rest number of training data %d\n', cas_number);
    end
end