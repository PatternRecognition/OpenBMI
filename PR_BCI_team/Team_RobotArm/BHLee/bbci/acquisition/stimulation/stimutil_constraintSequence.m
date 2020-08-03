function seq= stimutil_constraintSequence(nItems, nTrials, varargin)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'margin', 1);
                
nRounds= floor(nTrials/nItems);
seq= zeros(1, nTrials);

idx= 1:nItems;
for rr= 1:nRounds,
  satisfied= 0;
  while ~satisfied,
    tmp_seq= randperm(nItems);
    if idx(1)==1,
      satisfied= 1;
    else
      if isempty(intersect(tmp_seq(1:opt.margin), seq(idx(1)-[1:opt.margin]))),
        satisfied= 1;
      end
    end
  end
  seq(idx)= tmp_seq;
  idx= idx + nItems;
end

nLeftOvers= nTrials - nRounds*nItems;
tmp_seq= randperm(nItems);
seq(idx(1):nTrials)= tmp_seq(1:nLeftOvers);
