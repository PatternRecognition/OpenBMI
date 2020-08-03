function d = mahalanobis_dist(x,y,C)
% function d = mahalanobis_dist(x,y,C)
% calculate the mahalanobis distance of x and y.
% d = (x-y)'*C*(x-y)

% kraulem, 15/03/04
if nargin == 2
  % Euclidean Distance
  xmag = sum(x.*x,1);
  ymag = sum(y.*y,1);
  d = repmat(ymag, size(x,2), 1) + repmat(xmag', 1, size(y,2)) - 2*x'*y;
else
  % Mahalanobis Distance
  Cx = C*x;
  Cy = C*y;
  xmag = sum(x.*Cx,1);
  ymag = sum(y.*Cy,1);
  d = repmat(ymag, size(x,2), 1) + repmat(xmag', 1, size(y,2)) - 2*x'*Cy;
end

