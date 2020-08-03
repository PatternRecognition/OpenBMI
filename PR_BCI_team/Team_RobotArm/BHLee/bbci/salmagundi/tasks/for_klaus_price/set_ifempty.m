function opt= set_ifempty(opt, varargin)

nn= length(varargin)/2;
for n= 1:nn,
  fld= varargin{2*n-1};
  if ~isfield(opt,fld) | isempty(getfield(opt,fld)),
    opt= setfield(opt, fld, varargin{2*n});
  end
end
