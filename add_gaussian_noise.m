function add_gaussian_noise(cloud, shapename, variance)
  pts = cloud.Location;
  SAMPLING_SET = 200;
  srf = struct('X', pts(:, 1), 'Y', pts(:, 2), 'Z', pts(:, 3));
  % estimate diameter
  ifps = fps_euc(srf, SAMPLING_SET);
  Dfps = pdist2(pts(ifps, :), pts(ifps, :));
  diam = sqrt(max(Dfps(:)));

  sig = diam * variance;
  X = pts + randn(size(pts)) * sig;
  ply_filename = [shapename, '_gaussian_noise_', num2str(variance)];        
  pcwrite(pointCloud(X, 'Color', cloud.Color), [ply_filename, '.ply']);
  disp([ply_filename, '.ply saved.']);
end
