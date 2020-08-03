function imp= getModelParameterIndex(pClassy)
%imp= getModelParameterIndex(pClassy)

if isstruct(pClassy),
  if isfield(pClassy, 'param') & isfield(pClassy.param, 'index'),
    imp= [pClassy.param.index];
  else
    imp = getModelParameterIndex(pClassy.classy);
  end
  return;
end

if ~iscell(pClassy),
  imp= 2;
  return;
end

lClassy= length(pClassy);
imp= 1;
while imp<=lClassy & ...
    (~ischar(pClassy{imp}) | ...
    isempty(strmatch(pClassy{imp}, {'*lin','*log'}))),
  imp= imp+1;
end
