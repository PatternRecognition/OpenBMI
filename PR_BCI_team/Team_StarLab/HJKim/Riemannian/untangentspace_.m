function out = untangetspace(A, B)

% untangentspace

for i = 1:size(A, 3)
    out(:, :, i) = sqrtm(B) * expm(B ^ (-1 / 2) * A(:, :, i) * B ^ (-1 / 2)) * sqrtm(B);
end