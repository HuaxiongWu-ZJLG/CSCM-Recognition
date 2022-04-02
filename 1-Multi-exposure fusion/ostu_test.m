%Otsu?? 
clc; 
clear all; 
I=imread('.\one\auto2.jpg'); 
I=rgb2gray(I); 
I=double(I); 
figure,imshow(uint8(I)),title('ori'); 
[m,n]=size(I); 
Th=Otsu(I); 
Th 
for i=1:m 
    for j=1:n 
        if I(i,j)>=Th 
            I(i,j)=255; 
        else 
            I(i,j)=0; 
        end 
    end 
end 
figure,imshow(I),title('Otsu');
imwrite(I,'auto2.jpg');
 