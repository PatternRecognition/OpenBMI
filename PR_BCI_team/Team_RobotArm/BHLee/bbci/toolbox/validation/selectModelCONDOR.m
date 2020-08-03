function [classy, E, S, P] = selectModelCONDOR(dat, model, xTrials, ...
					       state, condor_ops)
%SELECTMODEL Finds good parameter for model selection
% For a classification of labeled datas dependent of some parameters 
% (e. g. SVM, regularised Fisher Discriminant, RDA, ...) this
% program search for good parameters.
% The Programm use the programs given by Benjamin Blankertz. Especially 
% doXvalidationPlus, sampleDivision and some nice utils like
% meshall are used.
%
% INPUT: dat: structure with the data (x) and the labels (y)(usual
%             format). No control of the dimensionality of the
%             fields. This  must do the model.program
%        model: two possibilities: 
%               1. no struct: name of the classification method and
%                             possible fix values for the call of
%                             the classifier (cell array)
%               2. struct with the following entries:
%                    inflvar: value that describe how important the
%                             variance is (default 0)
%                    seeforvar: range in which parameters will be
%                               not ignored dependent to their
%                               errors (default 0)
%                    program: a program in the format program(dat,model,  
%                             xTrials, state) like doXvalidationPlus
%                             (default doXvalidationPlus)
%                    classy: only classname or a cell array with fix
%                            values for the call of the classifier,
%                            or with the strings '*lin', '*log' in
%                            combination with the case where
%                            model.param is no struct (see there)
%                    msTrials: two/threedimensional vector for the cross 
%                              validation. See sampleDivisions.
%                              ONLY USED IF xTrials has no value
%                              (default  [3 10])
%                    msDepth:  positive number: how much steps for
%                              depthsearch. default 1
%                    param: one possible use is as non struct, only
%                           values. On this parameters the best
%                           parameter will calculate. Scaling lin
%                           oder log(10) is given by model.classy
%                           on the right position. Params will add
%                           on the right place. (works only for
%                           one-dimensional parameters)
%                           Another use of param is as a struct (or
%                           struct array). Then the following
%                           entries are possible:
%                       index: the place where the params must be set
%                              in the call of the train of the classifier 
%                       value: the param the algorithm must see for
%                              (default 0.05:0.1:0.95 if lin or -4:4 if log)
%                       scale: lin or log (default lin)
%                       range: the range where params can be
%                              (default: [ 0 1]  for lin, [-10 10]
%                              for log (note 10^ for log)
%                   params: structure (intern use for recursion). 
%                           If this field exist, model.param will
%                           be ignored.
%                       index: a array with the order of indices
%                              where the params must be
%                       scale: cell array with lin and log (default lin)
%                       minim: array with minimal values (default 0
%                              if lin, -10 if log)
%                       maxim: array with maximal values (default 1
%                              if lin, 10 if log)
%                       param: cell array, in each cell is one
%                              parameter combination
%                       value: cell array with all possible values
%                              for each parameter
%        xTrials: two/three  dimensional vector like model.msTrials
%                 (default is model.msTrials and if this not exist
%                 [3 10]). if xTrials is defined model.msTrials
%                 isn't used. See sampleDivisions
%        state: three possibilities 
%               1. flag: 0 no display, 1 display (default 0).
%               2. double flag: first for show, second for same.
%               3. struct with   (for recursion call)
%                  show for display (default 1)
%                  same for the same sampleDivision (default 1)
%                  divTr = train sets
%                  divTe = test sets
%                  recurs = point of recursion
%                  er = the best error
%                  variance = cell array of variances ( for the
%                  case that param are equal)
%        condor_opts: condor options, see condorset. If the
%                  argument is omitted, default options are used.
%
%
% OUTPUT: classy is a cell array (or a string) of the classifier with the 
%         best fixed params (maybe one is choose)
%         E: the best error
%         V: the variance for the best error (to the choosen one)
%         P: the best params (to the choosen one)
%
% Authors: Guido Dornhege (Guido.Dornhege@first.fhg.de)
%          Pavel Laskov (Pavel.Laskov@first.fhg.de)
% $Id: selectModelCONDOR.m,v 1.2 2002/10/29 10:17:35 candy_cvs Exp $

% Initialize condor
if nargin<5
  condor_ops = condorset;
end

c = condor(condor_ops.max_num_jobs);
c = displaylevel(c,condor_ops.display_level);  
condor_dir = condor_ops.dir;
condor_base = condor_ops.base;
condor_size = condor_ops.size;


% Control the format of dat and the existence of model and state
if ~isstruct(dat) 
  error('the 1st argument must be a struct'); 
end
if ~isfield(dat,'x') 
  error('the first argument must contain the data under the field x'); 
end
if ~isfield(dat,'y') 
  error('the first argument must contain the labels under the field y'); 
end
if ~exist('model','var') 
  error('no model as 2nd argument is specified'); 
end

% defaults
program = 'doXvalidationPlus';
inflvar =0;
seeforvar =0;
show =1;
same =1;
msTrials = [3 10];
msDepth = 1;
classyindex = '*lin';
scale = 'lin';
er = [];
variance = [];
params = [];
minimlog = -10;
minimlin = 0;
maximlog = 10;
maximlin = 1;
linvalue = 0.05:0.1:0.95;
linrange =[0 1];
logvalue = -4:4;
logrange=[-10 10];

% xTrials
if ~exist('xTrials', 'var') | isempty(xTrials)
  if ~isstruct(model) | ~isfield(model,'msTrials') | ...
    	isempty(model.msTrials)
    xTrials = msTrials;
  else
    xTrials = model.msTrials;
  end
end


% control the format of state
if ~exist('state','var') 
  state.show = show;
  state.same = same;
  state.er=er;
  state.variance = variance;
elseif ~isstruct(state)
  if ~(size(state,1)==1) 
    error('the format of the 4th argument is [a b] or a struct'); 
  end
  show = state(1);
  if length(state)==1
    state.same = same;
  else
    state.same = state(2);
  end   
  state.show = show;
  state.er=er;
  state.variance = variance;   
elseif isstruct(state)     
  if ~isfield(state,'show') state.show = show; end 
  if ~isfield(state,'same') state.same = same; end
  if ~isfield(state,'er') state.er = er; end
  if ~isfield(state,'variance') state.variance = variance; end
end

if ~isfield(state,'divTr') & state.same
  [state.divTr, state.divTe] = sampleDivisions(dat.y, xTrials);
end


% model no struct 
if ~isstruct(model)
  classy = model; P=[];
  [E,S] = feval(program, dat, model, xTrials,state.show);
  return;
end

% model.classy no cell - array
if ~iscell(model.classy) model.classy={model.classy}; end

% default for other fields in model
if ~isfield(model,'msTrials') | isempty(model.msTrials) 
  model.msTrials = msTrials; 
end
if ~isfield(model,'msDepth') | isempty(model.msDepth) 
  model.msDepth = msDepth; 
end
if ~isfield(model,'program') | isempty(model.program) 
  model.program = program; 
end
if ~isfield(model,'inflvar') | isempty(model.inflvar) 
  model.inflvar = inflvar; 
end
if ~isfield(model,'seeforvar') | isempty(model.seeforvar) 
  model.seeforvar = seeforvar; 
end
    
% if no model.params exist , a lot is to do

if ~isfield(model,'params') | isempty(model.params)

  % param no struct
  if ~isfield(model,'param') 
    error('you must specifiy param as field in the second argument'); 
  end
  if ~isstruct(model.param)
    V= model.param;
    model.param =[];
    model.param.index = getModelParameterIndex(model.classy);
    model.param.value = V;
    ip = length(model.param);
    for i=1:ip
      if model.param(i).index>length(model.classy) | ...
	isempty(model.classy{model.param(i).index}) 
	model.classy{model.param.index} = classyindex;
      end
      switch(model.classy{model.param(i).index})
       case '*log'
	model.param(i).scale = 'log';
       case '*lin'
	model.param(i).scale = 'lin';
	otherwise
	error(['the format for model.param.scale is not known.' ...
	       'Only log and lin possible']);
      end
    end
  end
  
  % other default for param
  for ip = 1:length(model.param)
    if ~isfield(model.param(ip),'index') | isempty(model.param(ip).index)
      error('no index in model.param is specified');
    end
    if ~isfield(model.param(ip),'scale') | isempty(model.param(ip).scale)
      model.param(ip).scale = scale;
    end
    switch(model.param(ip).scale)
     case 'lin'
      if ~isfield(model.param(ip), 'value') | ...
	     isempty(model.param(ip).value),
	model.param(ip).value = linvalue;
      end
      if ~isfield(model.param(ip),'range') | ...
	     isempty(model.param(ip).range)
	model.param(ip).range =linrange;
      end
     case 'log'
      if ~isfield(model.param(ip), 'value') | ...
	     isempty(model.param(ip).value),
	model.param(ip).value = logvalue;
      end
      if ~isfield(model.param(ip),'range') | ...
	     isempty(model.param(ip).range)
	model.param(ip).range =logrange;
      end
     otherwise
      error ('unknown scale');
    end
  end
  
  %translate in the right format (params)
  
  for ip=1:length(model.param)
    model.params.index(ip) = model.param(ip).index;
    model.params.scale{ip} = model.param(ip).scale;
    model.params.minim(ip) = model.param(ip).range(1);
    model.params.maxim(ip) = model.param(ip).range(2);
    model.params.value{ip} = sort(model.param(ip).value);
  end
  mesha = meshall(model.param(:).value);
  nVV = prod(size(mesha{1}));
  for i = 1:nVV
    for j=1:length(model.params.index)
      model.params.param{i}(j) = mesha{j}(i);    
    end
  end
elseif ~isfield(state,'recurs') | isempty(state.recurs)
  % default for model.params
  n = length(model.params.index);
  for i= 1:n
    if ~isfield(model.params,'scale') | isempty(model.params.scale{i})
      model.params.scale{i} = scale;
    end
    if ~isfield(model.params,'minim') | isempty(model.params.minim(i))
      if strcmp(model.params.scale{i},'log')
	model.params.minim(i) = minimlog;
      else
	model.params.maxim(i) = minilin;
      end
    end
    if ~isfield(model.params,'maxim') | isempty(model.params.maxim(i))
      if strcmp(model.params.scale{i},'log')
	model.params.minim(i) = maximlog;
      else
	model.params.maxim(i) = maxilin;
      end
    end
  end
  if ~isfield(model.params,'param') | isempty(model.params.param)
    error('model.params.param must be specified'); 
  end
  if ~isfield(model.params,'value') | isempty(model.params.value)
    for i= 1:n
      model.params.value{i}(1) = model.params.param{1}(i);
      for ip =2:length(model.params.param)
        c = find(model.params.param{ip}(i)<=model.params.value{i}(:));
        if isempty(c) 
	  d=1;
        else 
	  d = max(c);
        end
        if ~(model.params.param{ip}(i) == model.params.value{i}(d))
	  for l=length(model.params.value{i}):-1:1
	    model.params.value{i}(l+1) = model.params.value{i}(l);
	  end
	  model.params.value{i}(d) = model.params.param{ip}(i);
        end
      end
    end
  end
end

% if state.same divTr, divTe must set in dat
if state.same  & (~isfield(state,'divTr') | isempty(state.divTr))  
  dat.divTr = state.divTr;
  dat.divTe = state.divTe;
end

% for depthsearch state.recurs has to set
if model.msDepth>1 & (~isfield(state,'recurs') | isempty(state.recurs))
  state.recurs = 1;
end

% a lot of changes in model are needed, therefore we make a copy
modell = model;

% important variables
nCombos= length(model.params.param);
nParam = length(model.params.index);

% some nice output
if state.show
  if isfield(dat, 'title') & ~isfield(state,'recurs') 
    fprintf('model selection for %s using %s\n', dat.title, ...
            model.classy{1});
  end
  if isfield(state,'recurs') 
    fprintf('Depthsearch %d\n', state.recurs);
  end
  if ~isfield(state,'recurs') | state.recurs==1;
    fprintf('Format parameter: ');
    expStr = '';
    if strcmp(model.params.scale{1},'log') expStr = '10^'; end   
    fprintf('%d: %sP -> index %d', 1, expStr, ...
	    model.params.index(1));    
    for ip = 2:nParam
      expStr = '';
      if strcmp(model.params.scale{ip},'log') expStr = '10^'; end   
      fprintf(', %d: %sP -> index %d ', ip, expStr, ...
	      model.params.index(ip));
    end
    fprintf('\n');
  end
end

% no Xvalidation, calculate errors
job_ids = zeros(nCombos,1);
EE= zeros(nCombos,2);
SE= zeros(nCombos,2);
for ip= 1:nCombos,
  vv = '';
  for iv = 1:nParam
    if strcmpi(model.params.scale(iv), 'log'),
      v= 10^model.params.param{ip}(iv);
    else
      v= model.params.param{ip}(iv);
    end
    modell.classy{model.params.index(iv)}= v;
    vv = [vv, '_', num2str(v)];
  end
  if state.show,
    fprintf('parcomb%d= [%s]> ', ip, ...
	    vec2str(model.params.param{ip}, '%g'))
  end
  if state.same & ~isempty(state.er) & ~isempty(state.er{ip})
    EE(ip,:) = state.er{ip};
    SE(ip,:) = state.variance{ip};
    if state.show
      fprintf(['%4.1f' 177 '%.1f%%, [train: %4.1f' 177 '%.1f%%]' ...
	       '  (result above)\n'], ...
	      [EE(ip,:);SE(ip,:) ]);
    end
  else
    % Create structure with condor arguments;
    % The order must be the same as the order in function call!
    dat.fakeVar = vv;
    condor_args_in = struct('dat',dat,'model',{modell.classy},...
			    'xTrials',xTrials,'verbose',0);
    condor_args_out = {'E','C'};
    condor_args = struct('args_in', condor_args_in,...
			 'args_out', {condor_args_out});
    
    job = condor_job(condor_dir,condor_base, model.program,condor_args,condor_size,'dat.fakeVar');
    
    job = set(job,'isfunction','1');
    [c,result,job_ids(ip)] = schedule(c,job,'ignore-all');
    
    %[EE(ip,:), SE(ip,:)]= ...
    %	feval(model.program,dat, modell.classy, xTrials, state.show);
  end
end

time_tick = 60;
c = wait_until_done(c,time_tick);
res = foreach_history_job(c, 'return_args_out');
% Clean jobs: the data args file can be large; results files
% are not removed (see decription of 'all' in 'condor_jobs'.
%foreach_history_job(c,'clean_opt_file');
%foreach_history_job(c,inline('clean(job)'));
for ip = 1:length(res)
  idx = find(job_ids==res{ip}.job_id);
  EE(idx,:) = res{ip}.E;
  SE(idx,:) = res{ip}.C;
end


% Choose optimal configuration (maybe some equals)
[dummy,best] = min(EE(:,1)+model.inflvar*SE(:,1));
E = EE(best,1);
c = find(EE(:,1)<=(E+model.seeforvar*SE(best,1)));
if state.show,
  parStr= cell(1, nParam);
  for ip= 1:nParam, 
    expStr='';
    if strcmp(model.param(ip).scale,'log') expStr = '10^'; end 
    parStr{ip}= sprintf('%s%g', expStr, model.params.param{best}(ip));
  end
  fprintf(['minimum error (under consideration of the variance)', ...
	  ' was %.1f%% [%.1f%%] at <%s>\n\n'], ...
          EE(best,:), vec2str(parStr,'%s','/'));
end

% calculate now the datas belong to this result
classy = model.classy;
for ip=1:nParam
  if strcmpi(model.param(ip).scale, 'log'),
    v= 10^model.params.param{best}(ip);
  else
    v= model.params.param{best}(ip);
  end
  classy{model.params.index(ip)}= v;
  for k = 1:length(c)
    PA{k}(ip) = model.params.param{c(k)}(ip);
  end
  P(ip) = model.params.param{best}(ip);
end
for k=1:length(c)
  SA{k} = SE(c(k),:);
end
S = SE(best,:);


% msDepth = 1 (or E=0) get ready
if model.msDepth == 1 | E == 0        
  return;
end


%Depthsearch

%find interesting parameters

value=model.params.value;

%find all neighbors to the best parameters
for k=1:length(c)
  para{k}{1} = PA{k};
  for i=1:nParam
    pl = find(para{k}{1}(i) == model.params.value{i});
    if pl >1
      lef = model.params.value{i}(pl)-model.params.value{i}(pl-1);
    elseif model.params.value{i}(1)>model.params.minim(i)
      lef = (model.params.value{i}(pl)-model.params.minim(i));
      if state.recurs ==1 lef=2*lef; end % to see for the corners
    else 
      lef=[];
    end
    if pl < length(model.params.value{i})
      rig = model.params.value{i}(pl+1)-model.params.value{i}(pl);
    elseif model.params.value{i}(length(model.params.value{i})) ...
	   <model.params.maxim(i)
      rig = -(model.params.value{i}(pl)-model.params.maxim(i));
      if state.recurs ==1 rig=2*rig; end % to see for the corners
    else
      rig=[];
    end
    
   abc = length(para{k});
    if ~isempty(lef)
      for j = 1:abc
	para{k}{length(para{k})+1} = para{k}{j};
	para{k}{length(para{k})}(i) = para{k}{length(para{k})}(i)-0.5*lef;
	cc = find(para{k}{length(para{k})}(i) <=value{i}(:));
	if isempty(cc) 
	  value{i}(length(value{i})+1) = para{k}{length(para{k})}(i);
	elseif ~(para{k}{length(para{k})}(i) == ...
		 value{i}(min(cc)))
	  for m = length(value{i}):-1:(min(cc))
	    value{i}(m+1) = value{i}(m);
	  end
	  value{i}(min(cc)) = para{k}{length(para{k})}(i);	     
	end
      end
    end
    if ~isempty(rig)
      for j = 1:abc
	para{k}{length(para{k})+1} = para{k}{j};
	para{k}{length(para{k})}(i) = ...
	    para{k}{length(para{k})}(i)+0.5*rig;
	cc = find(para{k}{length(para{k})}(i) <= value{i}(:));
	if isempty(cc) 	
	  value{i}(length(value{i})+1) = para{k}{length(para{k})}(i);	     
	elseif ~(para{k}{length(para{k})}(i) == value{i}(min(cc)))
	  for m = length(value{i}):-1:min(cc)
	    value{i}(m+1) = value{i}(m);
	  end
	  value{i}(min(cc)) = para{k}{length(para{k})}(i);	     
	end	   
      end
    end
  end
end

% mesh all neighbors
pa=0;
for k=1:length(c)
  for j=1:length(para{k})
    flag=1;
    for i=1:pa
      if isequal(param{i}, para{k}{j})
	flag =0;
	if j==1
	  Er{i} = E;
	  Se{i} = SA{k};   
	end         
      end
    end
    if flag
      pa = pa+1;
      param{pa} = para{k}{j};         
      if j==1
	Er{pa} = E;
	Se{pa} = SA{k};   
      else Er{pa}=[];
	Se{pa} = [];
      end
    end
  end            
end

% all neighbors found, no prepare the recursion
model.msDepth = model.msDepth-1;
model.params.value = value;
model.params.param = param;
state.er = Er;
state.variance = Se;
state.recurs = state.recurs+1;
[classy,E,S,P] = selectModelCONDOR(dat,model,xTrials,state,condor_ops);

% now everything is done.






