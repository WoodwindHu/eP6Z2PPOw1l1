function [new_inter_patch, new_intra_patch] = findCorrespondingPoint(patch_s, patch_t, patch_r)
	[n, m] = size(patch_s);
	new_inter_patch = zeros(n,m);
	for i = 1:n
		dist_min = 999999;
		temp = 0;
		for j = 1:n
			dist_st = norm(patch_s(i,:) - patch_t(j,:));
			if dist_st < dist_min
				temp = j;
				dist_min = dist_st;
			end
		end
		new_inter_patch(i,:) = patch_t(temp,:);
	end
	[p, q] = size(patch_r);
	new_intra_patch = zeros(p,q);
	winr = p/n;
	for k = 1:winr % k patch
		for i = 1:n % for every point in patch_s
			dist_min = 999999;
			temp = 0;
			for j = 1:n % for every point in patch_k
				dist_st = norm(patch_s(i,:) - patch_r((k-1)*n+j,:));
				if dist_st < dist_min
					temp = (k-1)*n+j;
					dist_min = dist_st;
				end
			end
			new_intra_patch((k-1)*n+i,:) = patch_r(temp,:);
		end
	end
end