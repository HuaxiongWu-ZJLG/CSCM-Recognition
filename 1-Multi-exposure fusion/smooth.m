function Y=smooth(y,n)
m=length(y);
j=1;
for i=(n-1)/2+1:(m-(n-1)/2)
    p=i-(n-1)/2;
    q=i+(n-1)/2;
Y(j)=sum(y(p:q))/n;
% Y(j)=(sum(y(p:q))-y(n))/(n-1);
    j=j+1;
end
end



% [num]=xlsread('C:\Users\DEAN\Desktop\1.xlsx', 2)
% y1=num;  %原始数据加载
% y2=smooth(y1,9);
% plot(1:n,y2,'-o',1:n,y2,'-*')

