function s1= struct_sloppycat(s1, s2, varargin)

if isempty(s1),
  s1= s2;
  return;
elseif isempty(s2),
  return;
end

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...                 
    set_defaults(opt, ...
                 'dim', 2, ...
                 'keepfields', 3, ...
                 'matchsize', 0);

flds1= fieldnames(s1);
flds2= fieldnames(s2);

switch(opt.keepfields),
 case {0,'none'},
  flds= intersect(flds1, flds2);
  s1= rmfield(s1, setdiff(flds1, flds));
  s2= rmfield(s2, setdiff(flds2, flds));
 case {1,'first'},
  flds= flds1;
  if opt.matchsize,
    s2= struct_createfields(s2, setdiff(flds, flds2), 'matchsize',s1(1));
  else
    s2= struct_createfields(s2, setdiff(flds, flds2));
  end
 case {2,'last'},
  flds= flds2;
  if opt.matchsize,
    s1= struct_createfields(s1, setdiff(flds, flds1), 'matchsize',s2(1));
  else
    s1= struct_createfields(s1, setdiff(flds, flds1));
  end
 case {3,'all'},
  flds= union(flds1, flds2);
  if opt.matchsize,
    s1= struct_createfields(s1, setdiff(flds, flds1), 'matchsize',s2(1));
    s2= struct_createfields(s2, setdiff(flds, flds2), 'matchsize',s1(1));
  else
    s1= struct_createfields(s1, setdiff(flds, flds1));
    s2= struct_createfields(s2, setdiff(flds, flds2));
  end
 otherwise
  error('unknown value for OPT.keepfields');
end

s1= cat(opt.dim, s1, s2);
