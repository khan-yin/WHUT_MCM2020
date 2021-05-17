clc;
clear;
%% 
[num]=xlsread('point.xlsx');

m=num(:,1);
x=num(:,3);
y=num(:,4);
gtree=zeros(181,181);
cell=updatecell(x,y);
% for i=1:181
%     for j=1:181
%         cell(i,:)=[x(i),y(i)];
%     end
% end
fprintf("length(m):%d\n",length(m));
fprintf("length(x):%d\n",length(x));
fprintf("length(y):%d\n",length(y));
d=zeros(181,181);
d=distance(x,y);
% for i=1:181
%     for j=1:181
%         d(i,j)=sqrt((x(i)-x(j))^2+(y(i)-y(j))^2);
%         if (i==1&&j>=14) || (j==1&&i>=14)
%             d(i,j)=-1;
%         end
%     end
% end
% c=1;
% mintotaldis=inf;
% minc=0;
% minroute=0;
% minroute2=0;
% dlen=15;
% mintotaldis2=inf;
% mintotaldis1=0;
r =randperm(168)+13;%路径格式
temperature=1000;%初始化温度
cooling_rate=0.95;%温度下降比率
item=1;%用来控制降温的循环记录次数
ittt=[52,56]
iikk=setdiff(r,ittt);
v=[1,2,3,4,5,6,7,8,9,10,11,12,13,ittt,iikk];
xa=x(v);
ya=y(v);
kaka=d(v,v);
kaka=distance(xa,ya);
% for i=1:181
%     for j=1:181
%         kaka(i,j)=sqrt((xa(i)-xa(j))^2+(ya(i)-ya(j))^2);
%         if (i==1&&j>=16) || (j==1&&i>=16)
%             kaka(i,j)=-1;
%         end
%     end
% end
cell=updatecell(xa,ya);
% for i=1:181
%     for j=1:181
%         cell(i,:)=[xa(i),ya(i)];
%     end
% end
dlen=15;
[route,totaldis1]=Prime_one(kaka(1:dlen,1:dlen));
[route2,totaldis2]=Prime_two(kaka(2:181,2:181),route(2:dlen,2:dlen));
total=totaldis1+totaldis2;
fprintf("totaldistance1:%f,totaldistance2:%f,total=%f\n",totaldis1,totaldis2,total);
plotgraph(xa,ya,route,route2,cell,dlen);
% distance=totaldistance(route,dis);%计算路径总长度
% temperature_iterations = 1;
% % This is a flag used to plot the current route after 200 iterations
% plot_iterations = 1;
% 
% plotroute(city, route, distance,temperature);%画出路线

%% 
% while temperature>1.0 %循环条件
%     temp_route=change(route,'reverse');%产生扰动。分子序列变化
% %     fprintf("%d\n",temp_route(1));
%     temp_distance=totaldistance(temp_route,dis);%产生变化后的长度
%     dist=temp_distance-distance;%两个路径的差距
%     if(dist<0)||(rand < exp(-dist/(temperature)))
%         route=temp_route;
%         distance=temp_distance;
%         item=item+1;
%         temperature_iterations=temperature_iterations+1;
%         plot_iterations=plot_iterations+1;
%     end
%     if temperature_iterations>=10
%         temperature=cooling_rate*temperature;
%         temperature_iterations=0;
%     end
%     
%     if plot_iterations >= 20
%        plotroute(city, route, distance,temperature);%画出路线
%        plot_iterations = 0;
%     end
% %     fprintf("it=%d",item);
% end