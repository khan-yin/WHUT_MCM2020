clc;
%clear;
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

% route=Prime(d(2:181,2:181));
route=Prime(d(1:13,1:13));
for i=1:13
    for j=1:13
        gtree(i,j)=route(i,j);
    end
end

route2=Prime2(d(2:181,2:181),route(2:13,2:13));
for i=2:181
    for j=2:181
        gtree(i,j)=route2(i-1,j-1);
    end
end

%����·ͼ
scatter(x(1),y(1),'bo')  %���Ĺ�ˮվλ��
hold on;
scatter(x(2:13),y(2:13),'k*')  %һ����ˮվλ��
hold on;
scatter(x(14:181),y(14:181),'b.')   %������ˮվλ��
hold on;

xlabel('x')
ylabel('y')
% gplot(route,cell(2:181,:),'r-');
gplot(route,cell(1:13,:),'b-')
gplot(route2,cell(2:181,:),'r-')
for i=1:181%length(cell)
%     t=cell(i:181,:);
    text(x(i),y(i),[num2str(i)],'color','black','FontSize',10);
end
hleg=legend('���Ĺ�ˮվ','һ����ˮվ','������ˮվ','һ���ܵ�','�����ܵ�','Location','NorthEastOutside');
hold on;
