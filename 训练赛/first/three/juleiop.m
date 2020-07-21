clc
clear;
%% 
[num]=xlsread('point.xlsx');
m=num(:,1);
x=num(:,3);
y=num(:,4);
for i=1:181
    for j=1:181
        cell(i,:)=[x(i),y(i)];
    end
end
fprintf("length(m):%d\n",length(m));
fprintf("length(x):%d\n",length(x));
fprintf("length(y):%d\n",length(y));
d=zeros(181,181);
for i=1:181
    for j=1:181
        d(i,j)=sqrt((x(i)-x(j))^2+(y(i)-y(j))^2);
        if (i==1&&j>=14) || (j==1&&i>=14)
            d(i,j)=-1;
        end
    end
end

count=1;
mintotaldis=inf;
min_r=0;
min_cell=0;
number=2;
min_p=0;
min_v=0;
min_xa=0;
min_ya=0;
mintotaldis=inf;
min_r=0;
mincount=0;
minnumber=0;
number=2;
for number=1:11
    julei_res=[];
    max=nchoosek(11,number);
    for count=1:max
        isbad=0;
%         count=ceil(rand*max);%最大55
        P = zeros(1, 181);
        P(1,1) = 1;
        new_route=changeroute3(number,count);
        for i=1:length(new_route)
            P(1,new_route(i))=new_route(i);
        end
        dback=d;
        dback(dback<=0) = Inf;
        V = 1:181;
        V_P = V - P;
        p = P(P~=0);
        v = V_P(V_P~=0);
        rl=dback(p,v);
        [Y,index]=min(rl);
        vvp=p(index);
        julei_res(:,1)=vvp';
        julei_res(:,2)=v';
        vx=julei_res(:,1);
        vp=julei_res(:,2);
        list=zeros(181,50);
        for i=1:(13+number)
            ml=zeros(1,50);
            ml=gotlabel(new_route(i),vx,vp)';
            for j=1:length(ml)
                list(new_route(i),j)=ml(1,j);
            end
        end
        totaldis=0;
        for kkk=2:(13+number)
            k=list(new_route(kkk),:);
            k=k(k>0);
            v=[new_route(kkk),k];
            droute=d(v,v);
            [r,totaldistance,isbad]=Prime3(droute);
            if isbad==1
%                 fprintf("bad\n");
%                 isbad=1;
                break;
            end
            totaldis=totaldis+totaldistance;
            
        end
            t=ccl(new_route,number);
            totaldis=totaldis+t;
         if(isbad==0)%&&
             if totaldis<mintotaldis&&totaldis>0
                mintotaldis=totaldis;
                minnumber=number;
                mincount=count;
             end
             fprintf("count=%d\n",count);
             fprintf("totaldis:%f,minnumber:%d,mincount:%d\n",mintotaldis,minnumber,mincount);
         end
    end

    fprintf("number=%d\n",number);
    number=number+1;
end

% %画线路图
% scatter(x(1),y(1),'bo');  %中心供水站位置
% hold on;
% anser=p(2:length(min_p));
% scatter(x(anser),y(anser),'k*');  %一级供水站位置
% hold on;
% scatter(x(min_v),y(min_v),'b.');   %二级供水站位置
% hold on;
% xlabel('x');
% ylabel('y');
% xa=x(v);
% ya=y(v);
% %画线路图
% scatter(x(1),y(1),'bo')  %中心供水站位置
% hold on;
% anser=p(2:length(p));
% scatter(x(anser),y(anser),'k*')  %一级供水站位置
% hold on;
% scatter(x(v),y(v),'b.')   %二级供水站位置
% hold on;
% xlabel('x');
% ylabel('y');
% for i=1:length(xa)
%     cell(i,:)=[xa(i),ya(i)];
% end
