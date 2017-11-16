function [alphas, weakLearners] = adaBoost(T,N, featuretbl, iimages, labels)
    %initialization
    %fn = length(featuretbl); %number of features
    [fn,~] = size(featuretbl);
    alphas = zeros([1,T]);
    weakLearners = double(zeros([3,T]));
    weights = double(ones([T,N]));
    weights(1,:) = weights(1,:)/sum(weights(1,:));
    %boosting
    for t = 1:T
        tic;
        fprintf('round %d\n', t);
        summation = sum(weights(t,:));
        weights(t,:) = weights(t,:)./summation;
        epsilons = zeros([1,fn]);
        theta_ts = zeros([1,fn]);
        p_ts     = zeros([1,fn]);
        predictionArray = zeros(fn,N);
        parfor j = 1:fn
            [theta_j, p_j] = bestLearner(weights(t,:),j,N,featuretbl, iimages);
            theta_ts(j) = theta_j;
            p_ts(j) = p_j;
            epsilon_j = 0;
            for image_index = 1:N
                if p_j * computeFeature(image_index, j,featuretbl,iimages) > theta_j
                    prediction = 1;
                else
                    prediction = -1;
                end
                predictionArray(j,image_index) = prediction;
                epsilon_j = epsilon_j + weights(t,image_index) * abs(prediction - labels(image_index));
            end
            epsilons(j) = epsilon_j;
        end
        [epsilon_t, index_t] = min(epsilons);
        theta_t = theta_ts(index_t);
        p_t = p_ts(index_t);
        weakLearners(1,t) = theta_t;
        weakLearners(2,t) = p_t;
        weakLearners(3,t) = index_t;
        alpha_t = 1/2 * (1 - epsilon_t)/epsilon_t;
        
        if epsilon_t < 0.01
            alpha_t = 10;
        end
        alphas(t) =alpha_t;
        z_t = 2 * sqrt(epsilon_t * (1 - epsilon_t));
        weights(t+1,:) = weights(t,:)./z_t .* exp(-alpha_t .* labels .* predictionArray(index_t,:));
        toc;
        x = 0;
        y = 0;
        curr_alphas = alphas(1:t);
        curr_weakLearners = weakLearners(:,1:t);
        thetas = zeros([1,N]);
        k = 1;
        for j = 1:N
            if labels(j) == -1
                continue
            end
            thetas(k) = strongLearner_helper(curr_alphas, curr_weakLearners,j,featuretbl,iimages);
            k = k+1;
        end
        thetas(k:N) = [];
        theta = min(thetas);
        pred = zeros(1,N);
        parfor i = 1:N
            if labels(i) == -1
                y = y + 1;
                predict = strongLearner(curr_alphas, curr_weakLearners,i,featuretbl,iimages,theta);
                pred(i) = predict;
                if predict == 1
                    x = x +1;
                end
            end
        end
        if x/y <= 0.3 || y == 0
            break;
        end
    end
    alphas(t+1:T) = [];
    weakLearners(:,t+1:T) = [];
end

