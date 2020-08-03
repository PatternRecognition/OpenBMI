function epo_out = proc_redefineClasses(epo,idx,classes)

if nargin<3
    error('Not enough input arguments')
end

if ~iscell(idx)
    idx = {idx};
end

epo_out = proc_selectEpochs(epo,idx{1});
epo_out.y = ones(1,size(epo_out.y,2));
epo_out.className = classes(1);

for c = 2:length(idx)
   epoc = proc_selectEpochs(epo,idx{c});
   epoc.y = ones(1,size(epoc.y,2));
   epoc.className = classes(c);
   epo_out = proc_appendEpochs(epo_out,epoc);
end
