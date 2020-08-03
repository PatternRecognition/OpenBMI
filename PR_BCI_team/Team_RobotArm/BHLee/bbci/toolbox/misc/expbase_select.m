function [expbase, idx]= expbase_select(expbase, varargin)
%[expbase, idx]= expbase_select(expbase, paradigm)
%[expbase, idx]= expbase_select(expbase, <opt>)
%
%Selects entries from a data base that matches specified fields.
%
% IN  expbase  - data base of EEG experiments, see expbase_generate
%     paradigm - 
%     opt
%      .paradigm, .date, .subject, .appendix - match those fields
%      .operation - this operation is used to combine the indices matching
%                   the different fields, default 'intersect'.
%
% OUT expbase  - data base with selected entries only
%     idx      - indices of selected entries
%
% example
%  guido_base= expbase_select(expbase, 'subject','Guido', 'paradigm','imag*')
%  iml_base= expbase_select(expbase, {'imag_lett','imag_move'})

%% bb ida.first.fhg.de 07/2004

if length(varargin)==1,
  opt= struct('paradigm', varargin{1});
else
  opt= propertylist2struct(varargin{:});
end
opt= set_defaults(opt, ...
                  'operation', 'intersect');

if strcmp(opt.operation, 'union'),
  idx= [];
else
  idx= 1:length(expbase);
end

select_fields= intersect(fieldnames(expbase), fieldnames(opt));
for ff= 1:length(select_fields),
  fld= select_fields{ff};
  eval(['expcell= {expbase.' fld '};']);  %% how to do this correctly?
  pat= getfield(opt, fld);
  if iscell(pat),
    new_idx= [];
    for cc= 1:length(pat),
      ni= strpatternmatch(pat{cc}, expcell);
      new_idx= union(new_idx, ni);
    end
  else
    new_idx= strpatternmatch(pat, expcell);
  end
  idx= feval(opt.operation, idx, new_idx);
end
expbase= expbase(idx);
