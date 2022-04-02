function T=Otsu(I) 
[m,n]=size(I); 
I=double(I); 
count=zeros(256,1);  
pcount=zeros(256,1); 
for i=1:m 
    for j=1:n 
        pixel=I(i,j); 
        count(pixel+1)=count(pixel+1)+1; 
    end 
end 
dw=0; 
for i=0:255 
    pcount(i+1)=count(i+1)/(m*n); 
    dw=dw+i*pcount(i+1); 
end 
Th=0; 
Thbest=0; 
dfc=0; 
dfcmax=0; 
while (Th>=0 & Th<=255) 
    dp1=0; 
    dw1=0; 
    for i=0:Th 
        dp1=dp1+pcount(i+1); 
        dw1=dw1+i*pcount(i+1); 
    end 
    if dp1>0 
        dw1=dw1/dp1; 
    end 
    dp2=0; 
    dw2=0; 
    for i=Th+1:255 
        dp2=dp2+pcount(i+1); 
        dw2=dw2+i*pcount(i+1); 
    end 
    if dp2>0 
        dw2=dw2/dp2; 
    end 
    dfc=dp1*(dw1-dw)^2+dp2*(dw2-dw)^2; 
    if dfc>=dfcmax 
        dfcmax=dfc; 
        Thbest=Th; 
    end 
    Th=Th+1; 
end 
T=Thbest;