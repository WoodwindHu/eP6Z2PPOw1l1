function [new_inter_patch, new_intra_patch] = findCorrespondingPoint3(patch_c, patch_inter, patch_intra, pt_inter, pt_intra)
	pk = length(patch_c);
    new_inter_patch = zeros(1,pk);
    pcs = pointCloud(pt_intra.Location(patch_c,:));
    pct = pointCloud(pt_inter.Location(patch_inter,:));
    tform = pcregistericp(pcs ,pct);
    pcsn=pctransform(pcs,tform);
    for i = 1:pk
        [indices,~] = findNearestNeighbors(pct,pcsn.Location(i,:),1);
        new_inter_patch(i) = patch_inter(indices);
    end 
    [p, q] = size(patch_intra);
	new_intra_patch = zeros(p,q);
    for k = 1:p % p patch
        pct = pointCloud(pt_intra.Location(patch_intra(k,:),:));
        tform = pcregistericp(pcs ,pct);
        pcsn=pctransform(pcs,tform);
        for i = 1:pk
            [indices,~] = findNearestNeighbors(pct,pcsn.Location(i,:),1);
            new_intra_patch(k,i) = patch_intra(k, indices);
        end
    end
end
    
    
    
    
    