function t = ccl(new_route,number)
[num]=xlsread('point.xlsx');
m=num(:,1);
x=num(:,3);
y=num(:,4);
for i=1:181
    for j=1:181
        cell(i,:)=[x(i),y(i)];
    end
end
% fprintf("length(m):%d\n",length(m));
% fprintf("length(x):%d\n",length(x));
% fprintf("length(y):%d\n",length(y));
d=zeros(181,181);
for i=1:181
    for j=1:181
        d(i,j)=sqrt((x(i)-x(j))^2+(y(i)-y(j))^2);
        if (i==1&&j>=14) || (j==1&&i>=14)
            d(i,j)=-1;
        end
    end
end
r =randperm(168)+13;
ittt=setdiff(new_route,[1,2,3,4,5,6,7,8,9,10,11,12,13]);
iikk=setdiff(r,ittt);
v=[new_route,iikk];
xa=x(v);
ya=y(v);
kaka=d(v,v);
kaka=distance(xa,ya);

caccell=updatecell(xa,ya);

dlen=13+number;
[route_as,totaldis1]=Prime_one(kaka(1:dlen,1:dlen));
% gplot(route_as,caccell(1:dlen,:),'b-')
t=totaldis1;
end

