function output=imgaussfilt(I,sigma)
output=I;
ksize=double(uint8(3*sigma)*2+1);%窗口大小一半为3*sigma 
window = fspecial('gaussian', [1,ksize], sigma); %使用1行ksize列的高斯核对图像先进行x方向卷积，再进行y方向卷积
for i = 1:size(I,3)
    ret = imfilter(I(:,:,i),window,'replicate');
    ret = imfilter(ret,window','replicate');
    output(:,:,i) = ret;
end 
end