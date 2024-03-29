function [new_inter_patch, new_intra_patch] = findCorrespondingPoint2(patch_c, patch_inter, patch_intra, pt_inter, pt_intra)
	% input: order of patch
	pk = length(patch_c);
	new_inter_patch = zeros(1,pk);
	for i = 1:pk
		dist_min = 999999;
		temp = 0;
		for j = 1:pk
			dist_st = norm(pt_intra.Location(patch_c(i),:) - pt_inter.Location(patch_inter(j),:));
			if dist_st < dist_min
				temp = j;
				dist_min = dist_st;
			end
		end
		new_inter_patch(i) = patch_inter(temp);
	end
	[p, q] = size(patch_intra);
	new_intra_patch = zeros(p,q);
	for k = 1:p % p patch
		for i = 1:pk % for every point in patch_c
			dist_min = 999999;
			temp = 0;
			for j = 1:pk % for every point in patch_k
				dist_st = norm(pt_intra.Location(patch_c(i),:) - pt_intra.Location(patch_intra(k,j),:));
				if dist_st < dist_min
					temp = j;
					dist_min = dist_st;
				end
			end
			new_intra_patch(k,i) = patch_intra(k,temp);
		end
	end
end