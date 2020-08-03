function out= catifnonequal(in, dim)

if length(in)==1,
  out= in{1};
  return;
end

iseq= zeros(1, length(in)-1);
for ii= 1:length(in)-1,
  iseq(ii)= isequal(in{1}, in{ii});
end
isnum= apply_cellwise2(in, 'isnumeric');

if all(iseq),
  out= in{1};
else
  if all(isnum),
    if nargin<2,
      nd= apply_cellwise2(in, 'ndims');
      sz= ones(max(nd), length(in));
      for ii= 1:length(in),
        sz(:,ii)= size(in{ii});
      end
      dim= find([max(sz,[],2); 1]==1, 1, 'first');
    end
    out= cat(dim, in{:});
  else
    out= in;
  end
end
