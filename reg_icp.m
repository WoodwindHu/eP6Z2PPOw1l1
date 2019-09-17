%将选出的源块进行旋转平移，返回pn*3表示点的矩阵(包含/不包含空点两种)

%targetloc、sourceloc不含空点且局部，sxyz为起始坐标，n为块大小
%sn为含空点局部的变换后的源块矩阵，locsnv为不含空点非局部的变换后的源块矩阵,totalsnv为后者点数
function [sn,locsnv,totalsnv]=reg_icp(targetloc,sourceloc,sx,sy,sz,n)

%{
loct=target02pc.Location;
locs=source02pc.Location;
手工去掉最后两行点

loct(:,1)=loct(:,1)-220;
loct(:,2)=loct(:,2)-25;
loct(:,3)=loct(:,3)-85;

locs(:,1)=locs(:,1)-215;
locs(:,2)=locs(:,2)-25;
locs(:,3)=locs(:,3)-85;

pct=pointCloud(loct);
pcs=pointCloud(locs);

Rotation=pcregrigid(pct,pcs,'Extrapolate',true);
Rotation=invert(Rotation);
disp(Rotation.T);
pcsn=pctransform(pcs,Rotation);

figure(1)
pcshow(pct);
figure(2)
pcshow(pcs);
figure(3)
pcshow(pcsn);
%}

%旋转
pct=pointCloud(targetloc);
pcs=pointCloud(sourceloc);
Rotation=rotate_icp(pct,pcs);
%Rotation=invert(Rotation);
%disp(Rotation.T);
pcsn=pctransform(pcs,Rotation);

%voxelize
locsn=pcsn.Location;
%snvoxel=zeros(n,n,n,4);
snvoxelcount=zeros(n,n,n);

%去除超过patch大小的点存储
totalsn=size(locsn,1);
for i=1:totalsn
    tx=round(locsn(i,1));
    if tx<1||tx>n
        continue;
    end
    ty=round(locsn(i,2));
    if ty<1||ty>n
        continue;
    end
    tz=round(locsn(i,3));
    if tz<1||tz>n
        continue;
    end
    snvoxelcount(tx,ty,tz)=snvoxelcount(tx,ty,tz)+1;
    %snvoxel(tx,ty,tz,snvoxelcount(tx,ty,tz))=i;
end

%生成非局部不含空点坐标和局部含空点坐标
locsnv=zeros(1,3);
sn=zeros(1,3);
totalsnv=0;
t=0;
for i=1:n
    for j=1:n
        for k=1:n
            if snvoxelcount(i,j,k)==0
                t=t+1;
                sn(t,:)=[0,0,0];
            else
                t=t+1;                
                sn(t,:)=[i,j,k];
                totalsnv=totalsnv+1;
                locsnv(totalsnv,:)=[sx+i,sy+j,sz+k];
            end
        end
    end
end

%{
%查看点云
pcsn=pointCloud(locsnv);
figure(4)
pcshow(pcsn);
%}