%{
Copyright (c) 2015, Tom Mertens
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%}

%
% Implementation of Exposure Fusion
%
% written by Tom Mertens, Hasselt University, August 2007
% e-mail: tom.mertens@gmail.com
%
% This work is described in
%   "Exposure Fusion"
%   Tom Mertens, Jan Kautz and Frank Van Reeth
%   In Proceedings of Pacific Graphics 2007
%
%
% Usage:
%   result = exposure_fusion(I,m);
%   Arguments:
%     'I': represents a stack of N color images (at double
%       precision). Dimensions are (height x width x 3 x N).
%     'm': 3-tuple that controls the per-pixel measures. The elements 
%     control contrast, saturation and well-exposedness, respectively.
%
% Example:
%   'figure; imshow(exposure_fusion(I, [0 0 1]);'
%   This displays the fusion of the images in 'I' using only the well-exposedness
%   measure
%


function R = exposure_fusion(I,m)

r = size(I,1);
c = size(I,2);
N = size(I,4);

W = ones(r,c,N);

imgs=double(I)/255;

%compute the measures and combines them into a weight map
contrast_parm = m(1);
sat_parm = m(2);
wexp_parm = m(3);

%可视化Wcon，Wsa,Wep,权重
% Wcon = contrast(I(:,:,:,l));
% Wsa = saturation(I(:,:,:,l));
% Wep = well_exposedness(I(:,:,:,l));

% imwrite(contrast(I(:,:,:,1)),'Wcon1.jpg');
% imwrite(contrast(I(:,:,:,2)),'Wcon2.jpg');
% imwrite(contrast(I(:,:,:,3)),'Wcon3.jpg');
% imwrite(contrast(I(:,:,:,4)),'Wcon4.jpg');
% 
% imwrite(saturation(I(:,:,:,1)),'Wsa1.jpg');
% imwrite(saturation(I(:,:,:,2)),'Wsa2.jpg');
% imwrite(saturation(I(:,:,:,3)),'Wsa3.jpg');
% imwrite(saturation(I(:,:,:,4)),'Wsa4.jpg');
% 
% imwrite(well_exposedness(I(:,:,:,1)),'Wep1.jpg');
% imwrite(well_exposedness(I(:,:,:,2)),'Wep2.jpg');
% imwrite(well_exposedness(I(:,:,:,3)),'Wep3.jpg');
% imwrite(well_exposedness(I(:,:,:,4)),'Wep4.jpg');

if (contrast_parm > 0)
    W = W.*contrast(I).^contrast_parm;    
end
if (sat_parm > 0)
    W = W.*saturation(I).^sat_parm;
end
if (wexp_parm > 0)
    W = W.*well_exposedness(I).^wexp_parm;
end


% imwrite(W(:,:,1),'W1.jpg');
% imwrite(W(:,:,2),'W2.jpg');
% imwrite(W(:,:,3),'W3.jpg');
% imwrite(W(:,:,4),'W4.jpg');

%weight map refinement(USE recursive filter梯度滤波，细化权重矩阵)
for i=1:N
    W(:,:,i) = RF(W(:,:,i), 100, 4, 3, imgs(:,:,:,i));
end

% imwrite(W(:,:,1),'Rf1.jpg');
% imwrite(W(:,:,2),'Rf2.jpg');
% imwrite(W(:,:,3),'Rf3.jpg');
% imwrite(W(:,:,4),'Rf4.jpg');

%normalize weights: make sure that weights sum to one for each pixel
W = W + 1e-12; %avoids division by zero
W = W./repmat(sum(W,3),[1 1 N]);


% create empty pyramid
pyr = gaussian_pyramid(zeros(r,c,3));
nlev = length(pyr);

% multiresolution blending
for i = 1:N
    % construct pyramid from each input image
	pyrW = gaussian_pyramid(W(:,:,i));
	pyrI = laplacian_pyramid(I(:,:,:,i));
    
    % blend
    for l = 1:nlev
        w = repmat(pyrW{l},[1 1 3]);
        pyr{l} = pyr{l} + w.*pyrI{l};
    end
end
pyrI1 = laplacian_pyramid(I(:,:,:,1));
pyrI2 = laplacian_pyramid(I(:,:,:,2));
pyrI3 = laplacian_pyramid(I(:,:,:,3));
pyrI4 = laplacian_pyramid(I(:,:,:,4));
% imwrite(pyrI1{1},'pyrI1.jpg');
% imwrite(pyrI2{1},'pyrI2.jpg');
% imwrite(pyrI3{1},'pyrI3.jpg');
% imwrite(pyrI4{1},'pyrI4.jpg');

% imwrite(pyrI1{2},'pyrI12.jpg');
% imwrite(pyrI1{3},'pyrI13.jpg');
% imwrite(pyrI1{4},'pyrI14.jpg');
% imwrite(pyrI1{5},'pyrI15.jpg');

pyrW1 = gaussian_pyramid(W(:,:,1));
pyrW2 = gaussian_pyramid(W(:,:,2));
pyrW3 = gaussian_pyramid(W(:,:,3));
pyrW4 = gaussian_pyramid(W(:,:,4));
% imwrite(pyrW1{1},'pyrW1.jpg');
% imwrite(pyrW2{1},'pyrW2.jpg');
% imwrite(pyrW3{1},'pyrW3.jpg');
% imwrite(pyrW4{1},'pyrW4.jpg');

% imwrite(pyrW1{2},'pyrW12.jpg');
% imwrite(pyrW1{3},'pyrW13.jpg');
% imwrite(pyrW1{4},'pyrW14.jpg');
% imwrite(pyrW1{5},'pyrW15.jpg');

% reconstruct
R = reconstruct_laplacian_pyramid(pyr);
% imwrite(pyr{1},'R1.jpg');
% imwrite(pyr{2},'R2.jpg');
% imwrite(pyr{3},'R3.jpg');
% imwrite(pyr{4},'R4.jpg');
% imwrite(pyr{5},'R5.jpg');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% local contrast
function C = contrast(I)
N = size(I,4);

imgs=double(I)/255;
imgs_gray=zeros(size(I,1),size(I,2),N);
for i=1:N
    imgs_gray(:,:,i)=rgb2gray(imgs(:,:,:,i));
end
C=zeros(size(I,1),size(I,2),N);
% dense sift calculation
dsifts=zeros(size(I,1),size(I,2),32,N, 'single');
for i=1:N
    img=imgs_gray(:,:,i);
    ext_img=img_extend(img,16/2-1);
    [dsifts(:,:,:,i)] = DenseSIFT(ext_img, 16, 1);
end

for i=1:N
    C(:,:,i)=sum(dsifts(:,:,:,i),3);
end

% weighted-average (weighted_average==1) or winner-take-all (otherwise) 
% if weighted_average~=1
%     [x,labels]=max(C,[],3);
%     clear x;
%     for i=1:N
%         mono=zeros( size(I,1), size(I,2));
%         mono(labels==i)=1;
%         C(:,:,i)=mono;
%     end
% end
%weight-average/take all

% % contrast measure
% function C = contrast(I)
% h = [0 1 0; 1 -4 1; 0 1 0]; % laplacian filter
% N = size(I,4);
% C = zeros(size(I,1),size(I,2),N);
% for i = 1:N
%     mono = rgb2gray(I(:,:,:,i));
%     C(:,:,i) = abs(imfilter(mono,h,'replicate'));
% end

% saturation measure
function C = saturation(I)
N = size(I,4);
C = zeros(size(I,1),size(I,2),N);
for i = 1:N
    % saturation is computed as the standard deviation of the color channels
    R = I(:,:,1,i);
    G = I(:,:,2,i);
    B = I(:,:,3,i);
    mu = (R + G + B)/3;
    C(:,:,i) = sqrt(((R - mu).^2 + (G - mu).^2 + (B - mu).^2)/3);
end

% % 
% well-exposedness measure
% function C = well_exposedness(I)
% sig = .2;
% N = size(I,4);
% C = zeros(size(I,1),size(I,2),N);
% for i = 1:N
    % R = exp(-.5*(I(:,:,1,i) - .5).^2/sig.^2);
    % G = exp(-.5*(I(:,:,2,i) - .5).^2/sig.^2);
    % B = exp(-.5*(I(:,:,3,i) - .5).^2/sig.^2);
    % C(:,:,i) = R.*G.*B;
% end
function C = well_exposedness(I)
N = size(I,4);
I_gray=rgb2gray_n(I);
C=ones(size(I,1),size(I,2),N);
I_gray=double(I_gray);
C((I_gray>=0.90)|(I_gray<=0.1))=0;