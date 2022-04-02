clc
clear;

[num]=xlsread('C:\Users\DEAN\Desktop\1-M.xlsx',2); %加载excel数据

n1=length(num);
y1=num;  %原始数据加载

y2=smooth(y1,11); %处理后数据
n2=length(y2);
%%%转置矩阵y'，行变列
xlswrite('C:\Users\DEAN\Desktop\1-M.xlsx', y2',8); %保存到excel，（路径，数据，sheet，指定范围）
% figure;
% plot(1:n1,y1,'-o',1:n2,y2,'-*')
% figure;
% plot(1:n2,y2,'-o')