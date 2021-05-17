%% ==============��ȡ����==============
[xdata,textdata]=xlsread('data.xlsx'); %����20�����е����ݣ����ݰ��ձ����λ�ñ�����Excel�ļ�exp12_3_1.xls��
x_label=xdata(:,3); %������Ϊ������
y_label=xdata(:,4); %������Ϊ������
Demand=xdata(:,2);  %�ڶ���Ϊ������
C=[x_label y_label];      %�������
n=size(C,1);        %n��ʾ�ڵ㣨�ͻ�������

%�޸��±��
x_changed=zeros(n,1);
y_changed=zeros(n,1);
Demand_changed=zeros(n,1);
%�����±�ţ���������
for i=1:n
    x_changed(n-i+1,1)=x_label(i,1);
    y_changed(n-i+1,1)=y_label(i,1);
    Demand_changed(n-i+1,1)=Demand(i,1);
end
C=[x_changed y_changed];
%% =====�����ʻ�ͼ
figure(2)   %�����·��ͼ
tmp_1=[1,14,10,1,13,1,7,2,9,1,12,1,3,6,1];%4�ֳ�
tmp_2=[1,18,20,19,1,17,16,15,8,1,11,4,5,1];%6�ֳ�
plot([C(tmp_2,1)],[C(tmp_2,2)],'o-r');
hold on;
plot([C(tmp_1,1)],[C(tmp_1,2)],'o-b');
grid on

for i =1:size(C,1)
    text(C(i,1),C(i,2),['   ' num2str(i)]);
end
xlabel('�ͻ����ں�����'); ylabel('�ͻ�����������');
%xlswrite('shortestRoute.xlsx',Shortest_Route_1);
%xlswrite('shortestRoute.xlsx',Shortest_Route);