extend = 3;
size = 2;
step = 3;
featuretbl = zeros([1000000,8]);
index = 1;
for width = size:extend:64
    for height = size:extend:64
        for row = 1:step:64
            for col = 1:step:64
                if (64 - row) < height * 2
                    continue
                elseif (64 - col) < width
                    continue
                else
                    upperleft  = [row, col];
                    lowerright = [row + height, col + width];
                    upperleft_prime = [row + height, col];
                    lowerright_prime = [row + height * 2, col + width];
                    featuretbl(index,:) = [upperleft, upperleft_prime, lowerright, lowerright_prime];
                    index = index + 1;
                end
            end
        end
    end
end

for width = size:extend:64
    for height = size:extend:64
        for row = 1:step:64
            for col = 1:step:64
                if (64 - row) < height
                    continue
                elseif (64 - col) < width * 2
                    continue
                else
                    upperleft  = [row, col];
                    lowerright = [row + height, col + width];
                    upperleft_prime = [row, col+width];
                    lowerright_prime = [row + height, col + width*2];
                    featuretbl(index,:) = [upperleft, upperleft_prime, lowerright, lowerright_prime];
                    index = index + 1;
                end
            end
        end
    end
end
featuretbl(index:1000000,:) = [];
dlmwrite('featuretbl.txt', featuretbl);