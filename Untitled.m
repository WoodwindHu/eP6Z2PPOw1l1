I = eye(6);
% R = optimvar('R', 6, 6);
% F = rand(6, 6);
% prob = optimproblem;
% fi = rand(6,1);
% fj = rand(6,1);
% d_ij = 5;
% e = exp(1.0);
% R = I;
% F = -2 * e.^(-sum((I + 0.5 * (R - I)) * (fi - fj).^ 2)) * d_ij * (I + 0.5 * (R - I)) * (fi - fj) * (fi - fj)';
% F
% prob.Objective = sum(sum((R.*F)));
% prob.Constraints.cons1 = sum(sum(R.*I)) <= 5;
% 
% sol = solve(prob);
% sol.R


I = eye(6);
a = optimvar('a');
b = optimvar('b');
c = optimvar('c');
d = optimvar('d');
F = rand(2,2);
F(1,2) = -F(1,2);
F(2,2) = -F(2,2);
prob = optimproblem;
prob.Objective = a*F(1,1)+b*F(1,2)+c*F(2,1)+d*F(2,2);
prob.Constraints.cons1 = a+d <=4;
% prob.Constraints.cons2 = a>=0;
% prob.Constraints.cons3 = d>=0;
% prob.Constraints.cons4 = b>=0;
% prob.Constraints.cons5 = c>=0;

sol = solve(prob);
a