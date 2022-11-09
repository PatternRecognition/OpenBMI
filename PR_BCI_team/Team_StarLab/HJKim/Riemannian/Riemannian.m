function out = Rdis(A, B)

out = sqrt(sum(log(eig(A, B)) .^ 2));