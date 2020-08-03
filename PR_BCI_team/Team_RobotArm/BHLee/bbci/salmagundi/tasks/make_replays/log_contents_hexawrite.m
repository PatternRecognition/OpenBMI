function log_contents_hexawrite(number, varargin)

opt= propertylist2struct(varargin{:});
[opt, isdefault]= set_defaults(opt, ...
                               'code', 0, ...
                               'verbose', 0, ...
                               'fid', 1);

if nargin==0 | isempty(number),
  global LOG_DIR
  dd= dir([LOG_DIR 'feedback_hexawrite_*.log']);
  files= {dd.name};
  number= [];
  for ff= 1:length(files),
    nu= sscanf(files{ff}, 'feedback_hexawrite_%d.log');
    number= [number, nu];
  end
end

if length(number)>1,
  if ~isdefault.code,
    opt.code= 1;
  end
  for nn= 1:length(number),
    fprintf(opt.fid, '\nlogno= %d; ', number(nn));
    if opt.verbose,
      fprintf(opt.fid, '\n');
    end
    log_contents_hexawrite(number(nn), opt);
  end
  return;
end

[fb_opt,dum,init_file]= load_log('hexawrite', number);
fb_opt = set_defaults(fb_opt,...
                      'fs',25);
fb_opt.log= 0;

start= NaN;
writ= [];
writ_old= [];
written= {};
out= load_log;
while ~isempty(out)
  if iscell(out),
    if strcmp(out{1}, 'BLOCKTIME'),
      out= load_log;
      continue;
    end
    frameno= out{2};
    msec= frameno*1000/fb_opt.fs;
%    if ~opt.code,
    if opt.verbose,
      fprintf(1, '\r%010.3f ', msec/1000);
    end
    if out{4}==57 & strcmp(out{5}, 'String') & strcmp(out{6}, '3'),
      if opt.verbose,
        if opt.fid~=1, fprintf(opt.fid, '%010.3f ', msec/1000); end
        fprintf(opt.fid, ' -> countdown at 3\n');
      end
      start= msec;
    end
    if out{4}==57 & strcmp(out{5}, 'Visible') & strcmp(out{6}, 'off'),
      if opt.verbose,
        if opt.fid~=1, fprintf(opt.fid, '%010.3f ', msec/1000); end
        fprintf(opt.fid, ' -> countdown finished\n');
      end
      actual_start= msec;
      if isnan(start),
        start= msec;
      end
    end
    if out{4}==63 & strcmp(out{5}, 'String'),
      if isempty(writ),
%        start= msec;
      else
        if isempty(out{6}),
          if opt.fid~=1, fprintf(opt.fid, '%010.3f ', msec/1000); end
          fprintf(opt.fid, '\ntext reset\n');
%          keyboard
        end
      end
      stop= msec;
      if isempty(out{6}),
        writ= '';
      else
        writ= strcat(out{6});
      end
      if length(writ)==3 & strcmp(writ{1}, writ_old{2}),
        written= cat(2, written, writ_old(1));
      end
      if opt.verbose,
        if opt.fid~=1, fprintf(opt.fid, '%010.3f ', msec/1000); end
        if isempty(writ) | isempty(writ{1}),
          fprintf(opt.fid, ' ->\n');
        else
          fprintf(opt.fid, ' -> %s\n', strcat(writ{:}));
        end
      end
      writ_old= writ;
    end
  end
  out = load_log;
end
written= cat(2, written, writ);

if isempty(written) | (length(written)==1 & isempty(written{1})),
  fprintf(opt.fid, '%% EMPTY\n');
  return;
end

if opt.verbose,
  fprintf(opt.fid, '\nlogno= %d; ', number);
end
if opt.code,
  fprintf(opt.fid, 'opt.start= %.3f; opt.stop= %d;\n', ...
          start/1000, ceil(stop/1000+0.5));
  fprintf(opt.fid, 'replay(''hexawrite'', logno, opt, ...\n');
  fprintf(opt.fid, '       ''save'',sprintf(''%%s_hexawrite_%%03d'', sub_dir, logno));\n');
  fprintf(opt.fid, '%%%% ');
else
  fprintf(1, '\r                          \r');
  fprintf(opt.fid, '%010.3f: start\n', start/1000);
  fprintf(opt.fid, '%010.3f: stopp\n', stop/1000);
  fprintf(opt.fid, 'written: ');
end
fprintf(opt.fid, '%s\n', strcat(written{:}));
nc= length(strcat(written{:}));
ns= (stop-actual_start)/1000;
fprintf(opt.fid, ...
        '%%%% %d characters in %.1f sec: %.1f char/min.\n', ...
        nc, ns, nc/ns*60);
