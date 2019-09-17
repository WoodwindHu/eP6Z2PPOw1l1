%输入两组若干个控制点进行旋转平移，返回旋转矩阵

%pointst为目标块里的控制点，pointss为目标块里的控制点
function [Rotation]=rotate_icp(pointst,pointss)

divst=pointss-pointst;
addst=pointss+pointst;
pt=size(pointst,1);
A=zeros(4,4,3);
B=zeros(4,4);
for i=1:pt
    A(:,:,i)=[0 divst(i,:); ...
              divst(i,1) 0 0 0; ...
              divst(i,2) 0 0 0; ...
              divst(i,3) addst(i,:)];
    B=B+A(:,:,i)*A(:,:,i)';
end

[vec,val]=eig(B);
% [a,b,c,d]=vec(:,1);
a = vec(1,1);
b = vec(2,1);
c = vec(3,1);
d = vec(4,1);
Rotation=[a^2+b^2-c^2-d^2   2*b*c-2*a*d       2*b*d+2*a*c; ...
          2*b*c+2*a*d       a^2-b^2+c^2-d^2   2*c*d-2*a*b; ...
          2*b*d-2*a*c       2*c*d+2*a*b       a^2-b^2-c^2+d^2];

%{
pointst=[1 2 3; ...
         4 5 6; ...
         7 8 9];
     
pointss=[3 1 7; ...
         5 9 8; ...
         2 6 4];
%}