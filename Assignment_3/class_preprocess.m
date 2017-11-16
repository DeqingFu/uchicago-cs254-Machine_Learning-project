img = single(imread('class.jpg'));
img = img./255;
[r,c] = size(img);
class = zeros([r,c]);
step = 8;
class_iimages = zeros([100000,4096]);
coordinates = zeros([100000,2]);
count = 1;
for i = 1:step:(r-63)
    %fprintf('%d\n',int64(i/step)+1);
    for j = 1:step:(c-63)
        %(i,j) being the upper left corner
        coordinates(count,:) = [i,j];
        index = 1;
        for m = i:i+63
            for n = j:j+63
                val = 0.0;
                for k = i:m
                    val = val + sum(img(k,j:n));
                end
                class_iimages(count, index) = val;
                index = index + 1;
            end
        end
        if mod(count,100) == 0
            fprintf('%d\n',count);
        end
        count = count + 1;
    end
end
class_iimages(count:100000,:) = [];
coordinates(count:100000,:) = [];
dlmwrite('class_iimages.txt',class_iimages);
dlmwrite('coordinates.txt',coordinates);


