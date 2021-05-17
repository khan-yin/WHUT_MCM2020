function b = plotgraph(x,y,route,route2,cell,dlen)
%画线路图
scatter(x(1),y(1),'bo')  %中心供水站位置
hold on;
scatter(x(2:dlen),y(2:dlen),'k*')  %一级供水站位置
hold on;
scatter(x(dlen+1:181),y(dlen+1:181),'b.')   %二级供水站位置
hold on;

xlabel('x')
ylabel('y')
% gplot(route,cell(2:181,:),'r-');
gplot(route,cell(1:dlen,:),'b-')
gplot(route2,cell(2:181,:),'r-')
t=cell(1,:);
% for i=1:13%length(cell)
% %     t=cell(i:181,:);
%     text(x(i),y(i),[num2str(i)],'color','black','FontSize',10);
% end
    
hleg=legend('中心供水站','一级供水站','二级供水站','一级管道','二级管道','Location','NorthEastOutside');
hold on;
end

