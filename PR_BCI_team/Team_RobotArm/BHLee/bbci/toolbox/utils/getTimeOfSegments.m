function tim= getTimeOfSegments(mrk_orig, varargin)

opt= propertylist2struct(varargin{:});    
opt= set_defaults(opt, ...
                  'output', 'samples', ...
                  'fs', mrk_orig.fs);

ii= strmatch('New Segment', mrk_orig.type);
t= mrk_orig.time(ii);
dv= datevec(t, 'yyyymmddHHMMSS');

switch(opt.output),
 case 'datevec',
  tim= dv;
 case 'samples',
  tim= round(datenum(dv)*24*60*60*opt.fs);
  tim= tim-tim(1);
 case 'minutes',
  tim= datenum(dv)*24*60;
  tim= tim-tim(1);
 otherwise,
  error('output format unknown');
end
