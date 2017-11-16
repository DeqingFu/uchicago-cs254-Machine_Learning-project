function predict = prediction(cas_rounds,cascade_res,i,featuretbl,iimages)
    for j=1:cas_rounds
        T = int64(length(cascade_res{j})/4);
        alphas = cascade_res{j}(1:T);
        weakLearners = reshape(cascade_res{j}(T+1:T*4),[3,T]);
        predict =  predictStrongLearner(alphas, weakLearners,i,featuretbl,iimages,T);
        if predict == -1
            return;
        end
    end
end