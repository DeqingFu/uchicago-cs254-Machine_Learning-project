fprintf('loading data......\n');
coordinates  = dlmread('coordinates.txt');
class_iimages = dlmread('class_iimages.txt');
count = length(coordinates);
original = imread('class.jpg');

fprintf('processing......\n');
n = 0;
result = zeros(count,2);
for i = 1:count
    predict = prediction(cascades, res, i, featuretbl, class_iimages);
    if predict == 1
        n = n+1;
        coord = coordinates(i,:);
        row = coord(1);
        col = coord(2);
        result(n,:) = [row, col];
    end
end
result(n+1:count,:)= [];

new = zeros(count,2);
exist = zeros(1,n);
cnt = 0;
e_cnt = 0;
for i = 1:n
    coord_i = result(i,:);
    row_i = coord_i(1);
    col_i = coord_i(2);
    ret = [row_i, col_i];
    c = 1;
    if ismember(i,exist)
        continue
    end
    cnt = cnt+1;
    for j = i:n
        coord_j = result(j,:);
        row_j = coord_j(1);
        col_j = coord_j(2);
        if sqrt((row_i - row_j)^2 + (col_i - col_j)^2) <= 64 * sqrt(2)
            c = c + 1;
            ret(1) = ret(1) + row_j;
            ret(2) = ret(2) + col_j;
            e_cnt = e_cnt +1;
            exist(e_cnt) = j;
        end
    end
    ret = ret./c;
    new(cnt,:) = ret;
end
        
new(cnt+1:count,:)= [];

imshow(original);
hold on;
len = length(new);
for i = 1:len
    coord = new(i,:);
    row = coord(1);
    col = coord(2);
    r = rectangle('Position', [col,row,64,64]);
    r.EdgeColor = 'r';
    hold on;
end