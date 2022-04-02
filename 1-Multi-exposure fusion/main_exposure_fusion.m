%main_exposure_fusion()
% exposure fusion
I = load_images('E:\User\w\Desktop\3333');

L = GetFineDetail(I);   %提取出的图像细节
I_int = exposure_fusion(I, [1,1,1]);  %曝光融合图像
chi = 1;
I_out = I_int .* exp(repmat(chi*L, [1,1,3]));   
% figure(); imshow(L); title('fine details');
% figure(); imshow(I_int); title('intermedia image');
figure(); imshow(I_out); title('fusion result');
imwrite(I_out,'E:\User\w\Desktop\3333\R.jpg');

% imwrite(I_int,'mid.jpg');
% imwrite(L,'detail.jpg');


