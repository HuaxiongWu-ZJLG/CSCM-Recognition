clc
clear;

[num]=xlsread('C:\Users\DEAN\Desktop\1-M.xlsx',2); %����excel����

n1=length(num);
y1=num;  %ԭʼ���ݼ���

y2=smooth(y1,11); %���������
n2=length(y2);
%%%ת�þ���y'���б���
xlswrite('C:\Users\DEAN\Desktop\1-M.xlsx', y2',8); %���浽excel����·�������ݣ�sheet��ָ����Χ��
% figure;
% plot(1:n1,y1,'-o',1:n2,y2,'-*')
% figure;
% plot(1:n2,y2,'-o')