function [normals] = compute_normal(cloud, neighbor_count)
  count = cloud.Count;
  normals = zeros(count, 3);
  for i = 1 : count
    [indices, ~] = findNearestNeighbors(cloud, cloud.Location(i, :), neighbor_count);
    to_fit = cloud.Location(indices, :);
    [p, ~, ~] = regress(ones(neighbor_count, 1), to_fit);
    A = p(1);
    B = p(2);
    C = p(3);
    n = norm([A B C]);
    % Ax + By + Cz = 1
    normals(i, :) = [A B C] / n;
  end
end