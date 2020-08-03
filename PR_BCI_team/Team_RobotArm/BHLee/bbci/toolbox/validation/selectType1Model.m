function [classy, E, S, V]= selectType1Model(dat, model, xTrials, bound, state)
%[classy, errMean, errStd, P]= modelSelection(dat, model, xTrials, bound)
%
% IN   dat            - structure holding data (.x) and labels (.y)
%      model          structure describing the model
%           .classy   - classifier
%           .param    - parameter structure
%           .msTrials - cross validation trials for model selection
%           .msDepth  - for zooming into parameter space, default: 1
%
% OUT  classy  - classifier with selected parameters
%      errMean - test and training error for that classifer
%      errStd  - test and training error for that classifer
%      P       - selected model parameter value
%
% GLOBZ  NO_CLOCK
%
% still messy, must be totally rewritten!

if ~exist('state','var') | ~isstruct(state),
  if ~exist('state','var') | isequal(state,1),
    state.show=1; else state.show=0; 
  end
  if ~isstruct(model), 
    classy= model; V= [];
    [E, S]= doType1validation(dat, model, xTrials, bound, state.show);
    return; 
  end
  if ~isfield(model, 'msTrials'), model.msTrials= [3 10]; end
  if ~isfield(model, 'msDepth'), model.msDepth= 1; end
  if ~exist('xTrials','var'), xTrials= model.msTrials; end
  global NO_CLOCK
  state.clockMemo= NO_CLOCK;
  NO_CLOCK= 1;
  
  state.iter= 1;
  if length(model.msDepth)==1,
    model.msDepth= [model.msDepth 7];
  end
  state.V= [];
  state.S= [];
  state.E= [inf 0];
  [state.divTr, state.divTe]= sampleDivisions(dat.y, xTrials);
  if ~iscell(model.classy), model.classy= {model.classy}; end
  if ~isstruct(model.param),
    VV= model.param;
    model.param= [];
    model.param.index= getModelParameterIndex(model.classy);
    model.param.value= VV;
    if model.param.index>length(model.classy),
      model.classy{model.param.index}= '*lin';
    end
    if strcmp(model.classy{model.param.index}, '*log'),
      model.param.scale= 'log';
    elseif strcmp(model.classy{model.param.index}, '*lin+'),
      model.param.scale= 'lin+';
    end
  end
  for ip= 1:length(model.param),
    if ~isfield(model.param, 'scale') | isempty(model.param(ip).scale),
      model.param(ip).scale='lin';
    elseif strcmpi(model.param(ip).scale, 'lin+'),
      trainSize= length(state.divTr{1}{1});
      model.param(ip).minim= 1/sum(trainSize);
      model.param(ip).scale= 'lin';
    end
      
    switch(model.param(ip).scale),
      
     case 'lin',      
      model.param(ip).expStr= '';
      if ~isfield(model.param, 'minim') | isempty(model.param(ip).minim),
        model.param(ip).minim= 0;
      end
      if ~isfield(model.param, 'maxim') | isempty(model.param(ip).maxim),
        model.param(ip).maxim= 1;
      end
      if ~isfield(model.param, 'value') | isempty(model.param(ip).value),
        model.param(ip).value= linspace(0.05, 0.95, 10);
      end
      
     case 'log',      
      model.param(ip).expStr= '10^';
      if ~isfield(model.param, 'minim') | isempty(model.param(ip).minim),
        model.param(ip).minim= -10;
      end
      if ~isfield(model.param, 'maxim') | isempty(model.param(ip).maxim),
        model.param(ip).maxim= 10;
      end
      if ~isfield(model.param, 'value') | isempty(model.param(ip).value),
        model.param(ip).value= linspace(-4, 4, 9);
      end
      
     otherwise,
      error('unknown scale');
    end
  end
  state.classy= model.classy;
end


%NN= NN(find(NN>=state.minNN & NN<=state.maxNN));
nParam= length(model.param);
%VV= cell(1,nParam);
%[VV{:}]= ndgrid(model.param(:).value);  %% problem if nParam==1
VV= meshall(model.param(:).value);
nVV= prod(size(VV{1}(:)));
if state.show,
  if isfield(dat, 'title') & state.iter==1,
    fprintf('model selection for %s using %s\n', dat.title, ...
            model.classy{1});
  end
  for ip= 1:nParam,
    fprintf('par%d= %s[%s] -> index %d\n', ip, model.param(ip).expStr, ...
            vec2str(model.param(ip).value, '%g'), model.param(ip).index); 
  end
end
  
EE= zeros(nVV,2);
SE= zeros(nVV,2);
for iv= 1:nVV,
  for ip= 1:nParam,
    if strcmpi(model.param(ip).scale, 'log'),
      v= 10^VV{ip}(iv);
    else
      v= VV{ip}(iv);
    end
    state.classy{model.param(ip).index}= v;
  end
  if state.show,
    fprintf('%s> ', vec2str(vind2sub(size(VV{1}),iv)));
  end
  [EE(iv,:), SE(iv,:)]= ...
      doType1validation(dat, state.classy, xTrials, bound, state.show);
end

[E, best]= min(EE(:,1));
if state.show,
  parStr= cell(1, nParam);
  for ip= 1:nParam, 
    parStr{ip}= sprintf('%s%g', model.param(ip).expStr, VV{ip}(best));
  end
  fprintf('minimum error was %.1f%% [%.1f%%] at <%s>\n', ...
          EE(best,:), vec2str(parStr,'%s','/'));
end

if E<state.E(1),
  state.E= EE(best,:);
  state.S= SE(best,:);
  for ip= 1:nParam,
    state.V(ip)= VV{ip}(best);
  end
end
E= state.E;
S= state.S;
V= state.V;

iter= state.iter;
if state.iter<model.msDepth(1),
%  if state.show,
%    plot(NN, EE, '-o');
%    legend('test', 'train');
%    drawnow;
%  end
  NN= model.param(1).value;
  EEr= reshape(EE(:,1), size(VV{1}));
  NE= min(EEr(:,:), [], 2);         %% minimum errors at NN values
  be= mod(best, length(NE));
  worst= 45;
  if all(NE>worst),
    if state.show, 
      fprintf('not better than chance in this branch\n'); 
    end
  
  else
    if state.iter==1,
      tol= 0.25*SE(best,1);
    else
      tol= 0.125*SE(best,1);
    end
    upLimit= EE(best,1) + tol;
    nNN= length(NN);
    dNN= mean(diff(NN));
    topNi= max(find(NE<upLimit));
    botNi= min(find(NE<upLimit));
    border= 1 + (nNN>4);
    if be<=border & ~topNi>=nNN+1-border,
      topNi= max(2, min(ceil(nNN/3), topNi));
      topN= min(model.param(1).maxim, NN(topNi)+dNN);
      botN= max(model.param(1).minim, NN(1) - 0.5*(NN(end) - NN(1)) - dNN);
      newNN= linspace(botN, topN, model.msDepth(2));
      state.iter= state.iter+1;
    elseif be>=nNN+1-border & ~botNi<=border,
      botNi= min(nNN-1, max(floor(nNN*2/3), botNi));
      botN= max(model.param(1).minim, NN(botNi)-dNN);
      topN= min(model.param(1).maxim, NN(end) + 0.5*(NN(end) - NN(1)) + dNN);
      newNN= linspace(botN, topN, model.msDepth(2));
      state.iter= state.iter+1;
    else
      state.iter= state.iter+2;
      if botNi==1 & topNi==nNN,
        topNi= max(find(NE<EE(best,1)+tol/2));
        botNi= min(find(NE<EE(best,1)+tol/2));
      end
      if botNi==1 & topNi==nNN,
        if state.show, 
          fprintf('no improvement expected in this branch\n'); 
        end
        newNN= [];
      else
        topN= min(model.param(1).maxim, NN(topNi)+dNN);
        botN= max(model.param(1).minim, NN(botNi)-dNN);
        newNN= linspace(botN, topN, model.msDepth(2));
      end
    end
%%    newNN= setdiff(trunc(newNN,12), trunc(NN,12));  %% TODO
    if ~isempty(newNN),
      model.param(1).value= newNN;
      [c, E, S, V]= ...
          selectType1Model(dat, model, xTrials, bound, state);
    end
  end
end

if iter==1, %%isinf(state.iter),
  global NO_CLOCK
  NO_CLOCK= state.clockMemo;
  if state.show, 
    parStr= cell(1, nParam);
    for ip= 1:nParam, 
      parStr{ip}= sprintf('%s%g', model.param(ip).expStr, V(ip));
    end
    if model.msDepth>1,
      fprintf('overall minimum error was %.1f%% [%.1f%%] at <%s>\n', ...
              E, vec2str(parStr,'%s','/'));
    end

%    plot(state.NNN, state.EEE);
%    hold on;
%    plot(NN, EE, '.');
%    plot(NNN(winner), E, 'rx');
%    hold off;
%    legend('test', 'train');
%    drawnow;
  end
  
  classy= state.classy;
  for ip= 1:nParam,
    if strcmpi(model.param(ip).scale, 'log'),
      V(ip)= 10^V(ip);
    end
    classy{model.param(ip).index}= V(ip);
  end
end
