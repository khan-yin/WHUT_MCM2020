function b = plotgraph(x,y,route,route2,cell,dlen)
%����·ͼ
scatter(x(1),y(1),'bo')  %���Ĺ�ˮվλ��
hold on;
scatter(x(2:dlen),y(2:dlen),'k*')  %һ����ˮվλ��
hold on;
scatter(x(dlen+1:181),y(dlen+1:181),'b.')   %������ˮվλ��
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
    
hleg=legend('���Ĺ�ˮվ','һ����ˮվ','������ˮվ','һ���ܵ�','�����ܵ�','Location','NorthEastOutside');
hold on;
end

