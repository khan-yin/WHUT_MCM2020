%y1=y
%y2=y1'=y'
%y2'=y''
%y0=2,y1=0
function dy=F(t,y)
dy=zeros(2,1);
dy(1)=y(2);
dy(2)=1000*(1-y(1)^2)*y(2)-y(1);

