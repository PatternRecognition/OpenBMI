function ms= merge_structs(varargin)
%MERGE_STRUCTS - Merge fields of different struct into one struct
%
%Description:
%  The fields of a finite set of structs are merged into one struct.
%  When different structs have fields of one and the same name a
%  consistency check is perform. If the fields have the same value
%  everything is fine, otherwise an error is evoked.
%
%Usage:
%  S= merge_structs(S1, S2, <..., Sn>)
%
%Input:
%  Si: Structs to be merged
%
%Output:
%  S: Merged struct.
%
%Example:
%  s1= struct('a',1, 'b',2, 'c',3);
%  s2= struct('c',3, 'd',4);
%  s12= merge_structs(s1, s2)

% Author(s): Benjamin Blankertz, Jan 2005

ms= varargin{1};
szms= size(ms);
Flds= fieldnames(ms);
for ii= 2:nargin,
  if isempty(varargin{ii}),
    continue;
  end
  flds= fieldnames(varargin{ii});
  doub= intersect(flds, Flds);
  for jj= 1:length(doub),
    if ~isequal(getfield(ms,doub{jj}), getfield(varargin{ii},doub{jj})),
      error(sprintf('inconsistency in field <%s>.', doub{jj}));
    end
  end
  newfields= setdiff(flds, doub); newfields= newfields(:);
  if ~isempty(newfields),  %% Otherwise there will be an error in Flds= cat...
    for jj= 1:length(newfields),
      for kk= 1:prod(szms),
        ms= setfield(ms, {kk}, newfields{jj}, ...
                         getfield(varargin{ii}, {kk}, newfields{jj}));
      end
    end
    Flds= cat(1, Flds, newfields);
  end
end
