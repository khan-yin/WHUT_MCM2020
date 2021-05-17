clc;
% clear;
%% 
[num]=xlsread('point.xlsx');

m=num(:,1);
x=num(:,3);
y=num(:,4);
gtree=zeros(181,181);
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
c=1;
mintotaldis=inf;
minc=0;
minroute=0;
minroute2=0;
dlen=15;
mintotaldis2=inf;
mintotaldis1=0;
d15back=d(15,:);
dback=d;
% for i=14:181
%     for j=i+1:181
%         d=dback;
%         temp=d(i,:);
%         d(i,:)=d(14,:);
%         d(14,:)=temp;
% %         fprintf("totaldistance1:%f,totaldistance2:%f,total=%f\n",totaldis1,totaldis2,total);
%         temp=d(j,:);
%         d(j,:)=d(15,:);
%         d(15,:)=temp;
% %         if i==j
% %             j=j+1;
% %             continue;
% %         end
%         fprintf("第%d次\n",c);
%         [route,totaldis1]=Prime_one(d(1:dlen,1:dlen));
%         [route2,totaldis2]=Prime_two(d(2:181,2:181),route(2:dlen,2:dlen));
%         total=totaldis1+totaldis2;
%         if totaldis2<mintotaldis2
%             mintotaldis1=totaldis1;
%             minroute=route;
%             minroute2=route2;
%             mintotaldis2=totaldis2;
%             minc=c;
%             mintotaldis=total;
%             fprintf("totaldistance1:%f,totaldistance2:%f,total=%f\n",mintotaldis1,mintotaldis2,mintotaldis);
%         end
%         
%         j=j+1;
%         c=c+1;
%     end
%     d=dback;
%     temp=d(i,:);
%     d(i,:)=d(14,:);
%     d(14,:)=temp;
%     i=i+1;
% end
dlen=15;
[route,totaldis1]=Prime_one(d(1:dlen,1:dlen));
[route2,totaldis2]=Prime_two(d(2:181,2:181),route(2:dlen,2:dlen));
total=totaldis1+totaldis2;
fprintf("totaldistance1:%f,totaldistance2:%f,total=%f\n",totaldis1,totaldis2,total);
% fprintf("最终结果第%d次\n",minc);
% fprintf("totaldistance1:%f,totaldistance2:%f,total=%f\n",mintotaldis1,mintotaldis2,mintotaldis);

plotgraph(x,y,route,route2,cell,dlen);
