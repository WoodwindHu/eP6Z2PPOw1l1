function [new_inter_patch, new_intra_patch] = findCorrespondingPoint5(patch_c, patch_inter, patch_intra, F_inter, F_intra, patch_c_center, patch_intra_center, patch_inter_center)
	% input: order of patch
    % F_inter, F_intra: 6d features of points
    % patch_c_center(1*1), patch_inter_center(1*1), patch_intra_center(wink*1): order of
    % patch centers in inter- and intra-frame
    % patch centers
	pk = length(patch_c);
    fixed = F_intra(patch_c, 1:3) - repmat(F_intra(patch_c_center, 1:3), pk, 1); fixed_n = F_intra(patch_c_center,4:6)';
    % compute height field, and projection on reference plane
    hf = fixed*fixed_n; fixed_proj = fixed-hf*fixed_n';
    
    moving = F_inter(patch_inter, 1:3) - repmat(F_inter(patch_inter_center, 1:3), pk, 1);
    hm = moving*fixed_n; moving_proj = moving-hm*fixed_n';
    xx = sum((fixed_proj.^2),2);
    yy = sum((moving_proj.^2),2);
    xy = fixed_proj*moving_proj';
    DistMat = repmat(xx,1,length(yy))-2*xy+repmat(yy',length(xx),1);
    [~, idx] = min(DistMat,[],2);
    new_inter_patch = idx;
    
	[p, q] = size(patch_intra);
	new_intra_patch = zeros(p,q);
    for i = 1:p
        moving = F_intra(patch_intra(i,:), 1:3) - repmat(F_intra(patch_intra_center(i), 1:3), pk, 1);
        hm = moving*fixed_n; moving_proj = moving-hm*fixed_n';
        xx = sum((fixed_proj.^2),2);
        yy = sum((moving_proj.^2),2);
        xy = fixed_proj*moving_proj';
        DistMat = repmat(xx,1,length(yy))-2*xy+repmat(yy',length(xx),1);
        [~, idx] = min(DistMat,[],2);
        new_intra_patch(i,:) = idx;
    end
end