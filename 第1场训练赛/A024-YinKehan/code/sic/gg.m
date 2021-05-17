%  x=1:10:length(record)
%  
%  scatter(x,record(x));
%  hold on;
%  ylim([391 400 ])
%  scatter(890,record(890),'red');
% % while i<length(record)
% %     scatter(i,record(i));
% %     i=i+100;
% % end
sum(record([1:100]))/100;
ii=100;
aa=zeros(1,140002);

while ii<=14000
    aa(ii)=sum(record([ii-99:ii]))/100;
    ii=ii+100;
end

 ylim([395 400 ])
 hold on;
 plot([100:100:14000],aa([100:100:14000]))
 hold on;
 scatter([100:100:14000],aa([100:100:14000]))
 xlabel('每100次的均值')
 ylabel('均值')