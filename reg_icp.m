%��ѡ����Դ�������תƽ�ƣ�����pn*3��ʾ��ľ���(����/�������յ�����)

%targetloc��sourceloc�����յ��Ҿֲ���sxyzΪ��ʼ���꣬nΪ���С
%snΪ���յ�ֲ��ı任���Դ�����locsnvΪ�����յ�Ǿֲ��ı任���Դ�����,totalsnvΪ���ߵ���
function [sn,locsnv,totalsnv]=reg_icp(targetloc,sourceloc,sx,sy,sz,n)

%{
loct=target02pc.Location;
locs=source02pc.Location;
�ֹ�ȥ��������е�

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

%��ת
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

%ȥ������patch��С�ĵ�洢
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

%���ɷǾֲ������յ�����;ֲ����յ�����
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
%�鿴����
pcsn=pointCloud(locsnv);
figure(4)
pcshow(pcsn);
%}