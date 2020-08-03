function ca= cat_within_cell(dim, ca, cb)

szca= size(ca);
szcb= size(cb);
if ~isequal(szca, szcb),
  error('cell arrays must have the same size');
end
ca= reshape(ca, [1, prod(szca)]);
cb= reshape(cb, [1, prod(szcb)]);
for ii= 1:length(ca),
  ca{ii}= cat(dim, ca{ii}, cb{ii});
end
ca= reshape(ca, szca);
cb= reshape(cb, szcb);
