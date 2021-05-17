clc;
clear;
%% 
[num]=xlsread('point.xlsx');
i=0;
m=num(:,1);
x=num(:,3);
y=num(:,4);
gtree=zeros(181,181);
cell=updatecell(x,y);

fprintf("length(m):%d\n",length(m));
fprintf("length(x):%d\n",length(x));
fprintf("length(y):%d\n",length(y));
d=zeros(181,181);
d=distance(x,y);
dback=d;
% c=1;
% mintotaldis=inf;
% minc=0;
minroute1=0;
minroute2=0;
minx=0;
miny=0;
% dlen=15;
% mintotaldis2=inf;
% mintotaldis1=0;
r =randperm(168)+13;%路径格式
temperature=5000;%初始化温度
temperature_iterations=0;
cooling_rate=0.95;%温度下降比率
item=1;%用来控制降温的循环记录次数
route=[1,2,3,4,5,6,7,8,9,10,11,12,13,r];
x=x(route);
y=y(route);
d=d(route,route);
d=distance(x,y);
cell=updatecell(x,y);

dlen=15;
[minroute1,mintotaldis1]=Prime_one(d(1:dlen,1:dlen));
[minroute2,mintotaldis2]=Prime_two(d(2:181,2:181),minroute1(2:dlen,2:dlen));
mintotal=mintotaldis1+mintotaldis2;
fprintf("totaldistance1:%f,totaldistance2:%f,total=%f\n",mintotaldis1,mintotaldis2,mintotal);
plotgraph(x,y,minroute1,minroute2,cell,dlen);
% v=[1,2,3,4,5,6,7,8,9,10,11,12,13,r];
% xa=x(v);
% ya=y(v);
% kaka=d(v,v);
% kaka=distance(xa,ya);
% cell=updatecell(xa,ya);
% 
% dlen=15;
% [route,totaldis1]=Prime_one(kaka(1:dlen,1:dlen));
% [route2,totaldis2]=Prime_two(kaka(2:181,2:181),route(2:dlen,2:dlen));
% total=totaldis1+totaldis2;
% fprintf("totaldistance1:%f,totaldistance2:%f,total=%f\n",totaldis1,totaldis2,total);
% plotgraph(xa,ya,route,route2,cell,dlen);
record=[];
avg=[];
%% 
i=1;
while i<14028 %循环条件
    temp_route=changeroute(route,i);
    %temp_route=change(route,'swap');%产生扰动。分子序列变化
    xa=x(temp_route);
    ya=y(temp_route);
    da=d(temp_route,temp_route);
    da=distance(xa,ya);
    cella=updatecell(xa,ya);
    [route1,totaldis1]=Prime_one(da(1:dlen,1:dlen));
    [route2,totaldis2]=Prime_two(da(2:181,2:181),route1(2:dlen,2:dlen));
    total=totaldis1+totaldis2;
    record(i)=totaldis2;
    if mod(i,100)==0
        avg(i)=sum(total)/100;
    end
    dist=totaldis2-mintotaldis2;%两个路径的差距
    if dist<0
    %if(dist<0)||(rand < exp(-dist/(temperature)))
        route=temp_route;
        mintotaldis1=totaldis1;
        mintotaldis2=totaldis2;
        minroute1=route1;
        minroute2=route2;
        minx=xa;
        miny=ya;
        item=item+1;
        fprintf("%d\n",i);
        temperature_iterations=temperature_iterations+1;
    end
    i=i+1;
%     if temperature_iterations>=30
%         fprintf("temperature=%f item=%d\n\n",temperature,item);
%         temperature=cooling_rate*temperature;
%         temperature_iterations=0;
%     end
end

d=distance(minx,miny);
cell=updatecell(minx,miny);
fprintf("totaldistance1:%f,totaldistance2:%f,total=%f\n",mintotaldis1,mintotaldis2,mintotal);
plotgraph(minx,miny,minroute1,minroute2,cell,dlen);