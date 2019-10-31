function [R, weights, pr] = proximal_gradient_descent2(patch_s, patch_t, use_normal)
  if ~exist('use_normal', 'var')
    use_normal = true;
  end
  % initialize variables
  diff = patch_s(:, 1:3) - patch_t(:, 1:3);
  if use_normal
    norm_s = sum(patch_s(:, 4:6) .^ 2, 2) .^ 0.5;
    norm_t = sum(patch_t(:, 4:6) .^ 2, 2) .^ 0.5;
    cos_diff = sum(patch_s(:, 4:6) .* patch_t(:, 4:6), 2);
    cos_diff = 1.0 - abs(cos_diff ./ norm_s ./ norm_t);
    diff = [diff, cos_diff];
  end
  [node_dim, feat_dim] = size(diff);
  
  dist = sum(abs(patch_s(:, 1:3) - patch_t(:, 1:3)) .^ 2, 2);
  %dist = zeros(node_dim, 1);
  %for i = 1 : node_dim
  %  dist(i, 1) = norm(patch_s(i, 1:3) - patch_t(i, 1:3), 2) ^ 2;
  %end

  % D = diag(rand(feat_dim, 1));
  % U = orth(rand(feat_dim, feat_dim));
  % M = U' * D * U;
  R = eye(feat_dim);

%     weights = diag(exp(-(diff * M * diff')));
    weights = exp(-sum((diff * (R' * R)) .* diff, 2));
    cur_val = sum(weights .* dist);
    
%     weights = zeros(node_dim, 1);
%     cur_val = 0;
%     for i = 1 : node_dim
%       weights(i, 1) = exp(-(diff(i, :) * M * diff(i, :)'));
%       cur_val = cur_val + weights(i, 1) * dist(i, 1);
%     end
%     disp(['cur val:', num2str(cur_val)]);
    pr = [];
end

