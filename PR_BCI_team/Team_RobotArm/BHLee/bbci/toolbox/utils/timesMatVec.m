function X= timesMatVec(X, y)
%Z= timesMatVec(X, y)
%
% multiplies the vector y elementwise with each (column resp. row)
% vector of the matrix X

if min(size(y))>1, error('second argument must be a vector'); end

if size(y,2)==1,
  for ic= 1:size(X,2),
    X(:,ic)= X(:,ic) .* y;
  end
else
  for ir= 1:size(X,1),
    X(ir,:)= X(ir,:) .* y;
  end
end
