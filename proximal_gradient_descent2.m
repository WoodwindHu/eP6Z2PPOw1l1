function [R, weights, pr] = proximal_gradient_descent2(patch_s, patch_t, use_normal)
  if ~exist('use_normal', 'var')
    use_normal = true;
  end
  step = 1e-6;
  first = true;
  max_iter = 50;
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
  
  last_val = sum(exp(-sum((diff * (R' * R)) .* diff, 2)) .* dist);
%   last_val = sum(diag(exp(-(diff * M * diff'))) .* dist);
  
%   last_val = 0;
%   for i = 1 : node_dim
%     last_val = last_val + exp(-(diff(i, :) * M * diff(i, :)')) * dist(i, 1);
%   end
  pr = last_val;
  % calculate gradient
%   used = false;
%   h = waitbar(0, 'please wait');
  for iter = 1 : max_iter
%     str = ['Building graph... ', ...
%             num2str(roundn(iter * 100.0 / max_iter, -1)), '%'];
%     waitbar(iter / max_iter, h, str)
    gR = zeros(feat_dim);
%     coef = diag(exp(-(diff * M * diff'))) .* dist;
    coef = exp(-sum((diff * (R' * R)) .* diff, 2)) .* dist;
    for i = 1 : node_dim
%       grad = exp(-(diff(i, :) * M * diff(i, :)'));
%       grad = grad * dist(i, 1) * diff(i, :)' * diff(i, :);
%       gM = gM + grad;
      gR = gR + 2 * R * diff(i, :)' * diff(i, :) * coef(i);
    end
    gR = -gR;
    R = R - step * gR;

    % projection
    % [U, L] = eig(M);

    % if first
      % C = median(diag(L));
      % first = false;
    % end
    % L = max(L, 0);
    % L(L > C) = 0.1 * L(L > C);
    % M = real(U * L * U');

    C = 10;
    if trace(R) > C
      temp_val = (trace(R) - C) / feat_dim;
      R(logical(eye(size(R)))) = diag(R) - temp_val;
%       disp(['R>C!!! Now trace(R)=', num2str(trace(R))]);
    end

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
    if cur_val - last_val < 0 && abs(cur_val - last_val) < 1e-7
      disp(['Max iter: ', num2str(iter)]);
      break;
    end
    last_val = cur_val;
    pr = [pr; last_val];
  end
%   delete(h);
%   disp(['used:', num2str(used)]);
%   if iter >= max_iter
%     disp(['Max iter: ', num2str(max_iter)]);
%   end
end

