%w is the weight vector of the shape (1,N)
%f is the index of the feature
function [theta, p] = bestLearner(w,f,N,featuretbl, iimages)
    feature_values = zeros([1,N]);
    for i = 1:N
        feature_values(i) = computeFeature(i,f, featuretbl, iimages);
    end
    feature_values = sort(feature_values(1,:));
    epsilons = zeros([1,N]);
    ps = zeros([1,N]);
    for j = 1:N
        s_plus  = 0;
        s_minus = 0;
        t_plus  = 0;
        t_minus = 0;
        for k = 1:j
            if feature_values(k) > 0
                s_plus = s_plus + w(k);
            else
                s_minus = s_minus + w(k);
            end
        end
        for k = 1:N
            if feature_values(k) > 0
                t_plus = t_plus + w(k);
            else
                t_minus = t_minus + w(k);
            end
        end
        a = s_plus + (t_minus - s_minus);
        b = s_minus + (t_plus - s_plus);
        if a < b
            ps(j) = 1;
            epsilons(j) = a;
        else
            ps(j) = -1;
            epsilons(j) = b;
        end
    end
    [~, argmin] = min(epsilons);
    p = ps(argmin);
    if argmin == N
        theta = feature_values(argmin);
    else
        theta = double(feature_values(argmin) + feature_values(argmin+1))/2;
    end
end

