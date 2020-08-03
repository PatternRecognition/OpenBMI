function cpy= copy_struct(src, varargin)
%cpy= copy_struct(src, fld1, fld2, ...)
%cpy= copy_struct(src, 'not', fld1, fld2, ...)
%cpy= copy_struct(src, Flds)
%
% IN  src  - struct
%     fldx - name of fields of src
%     Flds - cell array {fld1, fld2, ...} or {'not', fld1, fld2, ...}
%            with fldx as above.
%
% OUT cpy  - copy of input struct (src), but containing 
%            :only the specified fields (variant 1)
%            :all but the specified fields (variant 2 with 'not')
%
% When you write 'NOT' with capital letters, a check is performed
% to make sure that the specified fields exist in the input struct src.
%
% NOTE: If input struct (src) contains a field with name 'not'
% you are in trouble.

%% bb ida.first.fhg.de 07/2004


cpy= [];
if length(varargin)==0,
  return;
elseif length(varargin)==1 & iscell(varargin{1}),
  flds= varargin{1};
else
  flds= varargin;
end

strict_checking= 0;
if strcmp(flds{1}, 'not'),
  yesorno= 0;
  flds= flds(2:end);
elseif strcmp(flds{1}, 'NOT'),
  yesorno= 0;
  flds= flds(2:end);
  strict_checking= 1;
else
  yesorno= 1;
end

fnames= fieldnames(src);
if strict_checking,
  fd= setdiff(flds, fnames);
 if ~isempty(fd),
   error(sprintf('field(s) %s not in struct', vec2str(fd)));
 end
end

for fi= 1:length(fnames),
  if xor(yesorno, isempty(strmatch(fnames{fi}, flds, 'exact'))),
    for k= 1:length(src),
      cpy= setfield(cpy, {k}, fnames{fi}, getfield(src, {k}, fnames{fi}));
    end
  end
end
