function out = tangentspace(A, B)

% tangentspace between A and B


for i = 1:size(A, 3)
    out(:, :, i) = sqrtm(B) * logm(B ^ (-1 / 2) * A(:, :, i) * B ^ (-1 / 2)) * sqrtm(B);
end

