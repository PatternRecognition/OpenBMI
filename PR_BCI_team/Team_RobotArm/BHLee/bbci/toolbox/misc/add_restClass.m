function mrk = add_restClass(mrk,tim,pos);
%ADD_RESTCLASS adds a rest Class to mrk
%
% usage:
%  mrk = add_restClass(mrk,<tim=1000,pos=500>);
%
% input:
%  mrk    a usual mrk structure with field trg (mrk structure with stimuli)
%  tim    in msec, if after a trg no response comes a rest class is used here
%  pos    time point after trg chosen for rest class
% output:
%  mrk    mrk structure with added rest class
%
% Guido DOrnhege, 02/09/04

if ~exist('pos','var') | isempty(pos)
  pos = 500;
end

if ~exist('tim','var') | isempty(tim)
  tim = 500;
end


pos = round(pos/1000*mrk.fs);
tim = round(tim/1000*mrk.fs);
posi = [];

for i = 1:length(mrk.trg.pos)
  in = find(mrk.pos>mrk.trg.pos(i) & mrk.pos<mrk.trg.pos(i)+tim);
  if isempty(in)
    posi = [posi,mrk.trg.pos(i)+pos];
  end
end

if ~isempty(posi)
  mrk.className{end+1} = 'relax';
  mrk.pos = cat(2,mrk.pos,posi);
  mrk.toe = cat(2,mrk.toe,zeros(1,length(posi)));
  mrk.y = cat(1,mrk.y,zeros(1,size(mrk.y,2)));
  mrk.y = cat(2,mrk.y,zeros(size(mrk.y,1),length(posi)));
  mrk.y(end,end-length(posi)+1:end) = 1;
  if isfield(mrk,'indexedByEpochs')
    for i = 1:length(mrk.indexedByEpochs)
      str = sprintf('mrk.%s = cat(2,mrk.%s,nan*ones(size(mrk.%s,1),%d));',mrk.indexedByEpochs{i},mrk.indexedByEpochs{i},mrk.indexedByEpochs{i},length(posi));
      eval(str);
    end  
  end
  
  [mrk.pos,ind] = sort(mrk.pos);
  mrk.toe = mrk.toe(ind);
  mrk.y = mrk.y(:,ind);
  if isfield(mrk,'indexedByEpochs')
    for i = 1:length(mrk.indexedByEpochs)
      str = sprintf('mrk.%s = mrk.%s(:,ind);',mrk.indexedByEpochs{i},mrk.indexedByEpochs{i});
      eval(str);
    end  
  end

end


    
    
    
