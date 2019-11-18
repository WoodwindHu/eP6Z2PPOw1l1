function [R, weights, pr] = proximal_gradient_descent4(patch_s, patch_t, use_normal)
  C = 4;
  if ~exist('use_normal', 'var')
    use_normal = true;
  end
  step = 0;
  flag = 1;
  first = true;
  max_iter = 5;
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
  R = eye(feat_dim);
  
  last_val = sum(exp(-sum((diff * (R' * R)) .* diff, 2)) .* dist);
  pr = last_val;
  % calculate gradient
  for iter = 1 : max_iter
    gR = zeros(4);
    coef = exp(-sum((diff*R').*(diff*R'), 2)) .* dist;
%     for i = 1 : node_dim
%       gR = gR + 2 * diff(i, :)' * diff(i, :) * R * coef(i);
%     end
    gR = 2* (repmat(coef, 1, feat_dim) .* diff)' * diff*R;
    gR = -gR;
    if step == 0
        step = 10^(-ceil(log10(abs(gR(1:1)))));
    end
    R = R - step * gR;
    for i = 1: feat_dim
        R(i,i)=max(R(i,i),0);
    end
    if trace(R) > C
      temp_val = trace(R) / C;
      R(logical(eye(size(R)))) = diag(R)/ temp_val;
      if flag ==1 
      disp(['R>C!!! Now trace(R)=', num2str(trace(R))]);
      flag = 0;
      end
    end

    weights = exp(-sum((diff*R').*(diff*R'), 2));
    cur_val = sum(weights .* dist);
    if cur_val - last_val < 0 && abs(cur_val - last_val) < 10
      disp(['Max iter: ', num2str(iter)]);
      break;
    end
    last_val = cur_val;
    pr = [pr; last_val];
  end
  histogram(weights);
  weights(weights>0.99) = 0;
end

