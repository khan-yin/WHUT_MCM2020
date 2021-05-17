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

[num]=xlsread('julei.xlsx');
vx=num(:,1);

vp=num(:,2);


c=2

list=zeros(13+c,50);
for i=2:(13+c)
    ml=gotlabel(i,vx,vp)';
    for j=1:length(ml)
        list(i,j)=ml(1,j);
    end
end
%画线路图
scatter(x(1),y(1),'bo')  %中心供水站位置
hold on;
scatter(x(2:13),y(2:13),'k*')  %一级供水站位置
hold on;
scatter(x(14:181),y(14:181),'b.')   %二级供水站位置
hold on;
count=1;
for kkk=2:13
    k=list(kkk,:);
    k=k(k>0);
    v=[kkk,k];
    droute=d(v,v);
    r=Prime3(droute);
    xa=x(v);
    ya=y(v);
    xlabel('x')
    ylabel('y')
    for i=1:length(xa)
        cell(i,:)=[xa(i),ya(i)];
        count=count+1;
    end
    gplot(r,cell,'r-')
end

