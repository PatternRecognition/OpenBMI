function out= bbci_prettyPrint(varargin)
%BBCI_PRETTYPRINT - Pretty print structs
%
%Synopsis:
%  bbci_prettyPrint(BBCI)
%  bbci_prettyPrint(FID, BBCI)
%  OUT= bbci_prettyPrint(...)
%
%Arguments:
%  BBCI - Struct to is to be printed
%  FID  - File handle(s) to which output is sent, default FID=1, i.e. screen.
%
%Returns:
%  OUT  - The output string

% 11-2011 Benjamin Blankertz


if isempty(varargin) || (length(varargin)==1 && isnumeric(varargin{1})),
  return
end

if isnumeric(varargin{1}),
  fid= varargin{1};
  varargin(1)= [];
else
  fid= 1;
end

bbci= varargin{1};

default_opt_str= struct('max_dim', 2, 'max_numel', 100);
opt= propertylist2struct(varargin{2:end});
opt= set_defaults(opt, ...
                  'prefix', '#', ...
                  'opt_str', default_opt_str);

out= '';
Flds= fieldnames(bbci);
for fi= 1:length(Flds),
  fld= Flds{fi};
  val= bbci.(fld);
  if isstruct(val),
    N= numel(bbci.(fld));
    for nn= 1:N,    
      val= bbci.(fld)(nn);
      SubFlds= fieldnames(val);
      for si= 1:length(SubFlds),
        subfld= SubFlds{si};
        subval= val.(subfld);
        str= sprintf('.%s = %s', subfld, toString(subval, opt.opt_str));
        if si==1,
          if N==1,
            str0= sprintf('%s', fld);
          else
            str0= sprintf('%s(%d)', fld, nn);
          end
          str= strcat(str0, str);
          blanks= repmat(' ', [1 length(str0)]);
        else
          str= [blanks, str];
        end
        out= sprintf('%s%s%s\n', out, opt.prefix, str);
      end
    end
  else
    str= sprintf('%s = %s', fld, toString(val, opt.opt_str));
    out= sprintf('%s%s%s\n', out, opt.prefix, str);
  end
end

bbci_log_write(fid, '%s', out);

if nargout==0,
  clear out
end
