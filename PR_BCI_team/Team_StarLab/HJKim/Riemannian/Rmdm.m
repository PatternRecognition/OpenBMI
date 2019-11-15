function out = Rmdm(A, B)

for i = 1:size(A, 3)
    for j = 1:size(B, 3)
        res(i, j) = Rdis(A(:, :, i), B(:, :, j));
    end
    out(i) = find(res(i, :) == min(res(i, :)));
end