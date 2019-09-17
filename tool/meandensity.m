function density = meandensity(A)

dM1 = pdist2(A,A,'euclidean','Smallest',2);
density = mean(dM1(2,:));
end