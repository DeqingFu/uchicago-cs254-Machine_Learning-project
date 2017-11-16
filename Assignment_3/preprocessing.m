cd('./faces');
output = double(zeros([4000,64*64]));
for i = 1:2000
    disp(i);
    input_img = imread(strcat('face',int2str(i-1),'.jpg'));
    img = double(rgb2gray(input_img))/255;
    for r = 1:64
        for c = 1:64
            val = 0.0;
            for j = 1:r
                for k = 1:c
                    val = val + img(j,k);
                end
            end
            output(i*2,64*(r-1)+c) = val;
        end
    end
end
cd('..');

cd('./background');
for i = 1:2000
    disp(i+2000);
    input_img = imread(strcat(int2str(i-1),'.jpg'));
    img = double(rgb2gray(input_img))/255;
    for r = 1:64
        for c = 1:64
            val = 0.0;
            for j = 1:r
                for k = 1:c
                    val = val + img(j,k);
                end
            end
            output(i*2-1,64*(r-1)+c) = val;
        end
    end
end
cd('..');
%For the iimages, 
%all with odd indices are nonfaces
%all with even indices are faces
dlmwrite("iimages.txt",output);

        