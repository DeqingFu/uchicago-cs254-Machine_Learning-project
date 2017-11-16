function summation = strongLearner_helper(alphas, weakLearners,i,featuretbl,iimages)
    T = length(alphas);
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
end