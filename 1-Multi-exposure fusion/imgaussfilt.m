function output=imgaussfilt(I,sigma)
output=I;
ksize=double(uint8(3*sigma)*2+1);%���ڴ�Сһ��Ϊ3*sigma 
window = fspecial('gaussian', [1,ksize], sigma); %ʹ��1��ksize�еĸ�˹�˶�ͼ���Ƚ���x���������ٽ���y������
for i = 1:size(I,3)
    ret = imfilter(I(:,:,i),window,'replicate');
    ret = imfilter(ret,window','replicate');
    output(:,:,i) = ret;
end 
end