function predict = predictStrongLearner(alphas, weakLearners,i,featuretbl,iimages,T)
    weak_predicts = zeros([1,T]);
    for t = 1:T
        theta_t = weakLearners(1,t);
        p_t = weakLearners(2,t);
        feature_t = weakLearners(3,t);
        weak_learn = p_t * (computeFeature(i, feature_t,featuretbl,iimages) - theta_t);
        if weak_learn > 0
            weak_predicts(t) = 1;
        else
            weak_predicts(t) = -1;
        end
    end
    summation = sum(alphas.*weak_predicts);
    if summation > 0
        predict = 1;
    else
        predict = -1;
    end
end