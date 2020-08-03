function eye = eye_readIDF(filename)
%
% Reads an SMI Eyetracking Data Text File (IDF-File) into Matlab.
%
% USAGE
%    function eye = eye_readIDF(filename)
%
% IN:   filename    -     filename or cell array of filenames
%
% OUT:  eye         -     data struct (with fields .x/.mrk/.header/.variables)
%
% Simon Scholler, 2011
%

global EYE_DAT_DIR


% parse input
if isstr(filename) && filename(end)=='*'
  single_file = 0;
  if ~isabsolutepath(filename)
      filename = [EYE_DAT_DIR filename];
  end
  files = struct2cell(dir(filename));
  filename = files(1,:);
elseif isstr(filename)
  filename = {filename};
  single_file = 1;
else
  single_file = 0;
end
  

for file = 1:length(filename)
  if ~isabsolutepath(filename{file})
      fid = fopen([EYE_DAT_DIR filename{file}]);
  else
      fid = fopen(filename{file});
  end

  % read IDF header
  header = '';
  stop = 0;
  while ~stop && ~feof(fid)
    fline = fgetl(fid);
    if length(fline)>3 && isequal('Time',fline(1:4))
      stop = 1;
      eye.variables{file} = cell_flaten(textscan(fline,'%s','Delimiter', '\t'));      
      eye.header{file} = header;
    else
      fline = strrep(fline,'\','\\');
      header = [header fline ' \n'];
    end
  end

  % format string
  nVars = length(eye.variables{file})-3;
  vars = repmat({' %f'},1,nVars);
  vars = cat(2,vars{:});
  data_format = ['%f %s %f' vars];
  mrk_format = '%f %s %f %[^\n]';
  
  % get number of lines in file
  s=char(textread(filename{file},'%[^\n]'));
  nLines = size(s,1);  
  
  c = textread(filename{file},'%[^\n]');
  idx_hdr = cellfun(@(x) ~isempty(strfind(x,'##')), c);
  
  % Store Header
  eye.header{file} = char(c(idx_hdr));
  
  % Store Sample Rate
  idx_fs = find(cellfun(@(x) ~isempty(strfind(x,'## Sample Rate:')), c));
  tmp = textscan(c{idx_fs},'%s');
  eye.fs{file} = str2double(tmp{1}{end});
  
  % Store Variable Names
  c = c(~idx_hdr);
  eye.variables{file} = cell_flaten(textscan(c{1},'%s','Delimiter', '\t'));    
  
  % Store Data & Triggers
  c = c(2:end);
  idx_mrk = find(cellfun(@(x) ~isempty(strfind(x,'# Message: SyncMarker')), c));  
  %idx_mrk = idx_mrk(1:end-3);  % HACK FOR TESTING PURPOSES ONLY
  idx_dat = find(cellfun(@(x) ~isempty(strfind(x,'SMP')), c));
  idx_dat = idx_dat( idx_dat>idx_mrk(1) & idx_dat<idx_mrk(end) );  % remove data before the first and after the last marker
  
  eye.dat{file} = cell(length(idx_dat),length(eye.variables{file}));
  for n = 1:length(idx_dat)
      eye.dat{file}(n,:) = textscan(c{idx_dat(n)},data_format);
  end

  eye.mrk{file}.desc = cell(length(idx_mrk),1);
  eye.mrk{file}.t = NaN(length(idx_mrk),1);  
  for n = 1:length(idx_mrk)
      line = textscan(c{idx_mrk(n)},mrk_format);
      eye.mrk{file}.desc{n} = line{end}{1};
      eye.mrk{file}.t(n) = line{1};
  end   
  
end

if single_file  % decell fields and add time fields
   eye.variables = eye.variables{1};
   eye.header = eye.header{1};
   eye.dat = eye.dat{1};  
   eye.mrk = eye.mrk{1};
   eye.fs = eye.fs{1};
   eye.t = (cell2mat(eye.dat(:,1))'-eye.mrk.t(1)) / 1000; %eye.fs;  % time in msec   
   eye.mrk.t = (eye.mrk.t-eye.mrk.t(1))' / 1000, % eye.fs;  % convert to msec and set first marker at t=0      
else  % merge fields and add time fields
   dat = eye.dat{1};
   mrk.desc = eye.mrk{1}.desc;
   mrk.t = (eye.mrk{1}.t-eye.mrk{1}.t(1))' / 1000 %eye.fs{1};  % convert to msec and set first marker at t=0   
   eye.t = (cell2mat(eye.dat{1}(:,1))'-eye.mrk{1}.t(1)) / 1000; %eye.fs{1};
   for f = 2:length(filename)
      if ~isequal(eye.variables{1},eye.variables{f})
          error('ET files have a different number of variables.')
      elseif ~isequal(eye.fs{1},eye.fs{f})
          error('Sampling frequencies of ET files differ.')
      end
      dat = [dat; eye.dat{f}];
      mrk.desc = [mrk.desc; eye.mrk{f}.desc];
      eye.t = [eye.t (cell2mat(eye.dat{f}(:,1))'-eye.mrk{f}.t(1))/1000 + eye.t(end) + 1000/eye.fs{f}];  % time in msec %eye.fs{f} + eye.t(end) + 1000/eye.fs{f}];  % time in msec
      eye.mrk{f}.t = (eye.mrk{f}.t-eye.mrk{f}.t(1))' / 1000; %eye.fs{f}; % convert to msec and set first marker at t=0  
      mrk.t = [mrk.t eye.mrk{f}.t+mrk.t(end)+1000/1000]; %eye.fs{f}];
   end
   eye.variables = eye.variables{1};
   eye.fs = eye.fs{1};   
   eye.dat = dat;
   eye.mrk = mrk;
end


