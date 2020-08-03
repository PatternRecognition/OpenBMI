function [Mrk,logf,flogf] = extract_logfiles(Mrk,file,opt);
%[MRK, LOGF, FLOGF] = extract_logfiles(MRK, FILE, OPT)
%
%OPT contains classifier (OPT.log_info) and feedback (OPT.log_fb) log for 
%*all* files of one whole session.
%This function extracts those parts of the logs, that correspond to the given
%FILE -> LOGF (classifier log) and FLOGF (feedback log).
%MRK is in some cases corrected, given the information of the log files.

% Mrk: markers from .vmrk-file.
global problem_marker_bit
if isempty(problem_marker_bit)
  problem_marker_bit = 0;
end

log_info = opt.log_info;
log_fb = opt.log_fb;

% check if the log_info-positions make sense.
% WARNING: this should actually be in the function that 
% delivers log_fb and log_info!
for ii = 1:length(log_info)
  problem_ind = find(diff(log_info(ii).mrk.pos)<0);
  if length(problem_ind)>0
    % something went wrong when concatenating logfiles.
    fprintf('fixing log_info(%i)\n',ii);
    for jj = 1:length(problem_ind)
      log_info(ii).mrk.pos(problem_ind(jj)+1:end) = log_info(ii).mrk.pos(problem_ind(jj)+1:end)+log_info(ii).mrk.pos(problem_ind(jj));
    end
    for kk = ii+1:length(log_info)
      % make the logfiles' positions increasing over all logfiles.
      log_info(kk).mrk.pos = log_info(kk).mrk.pos+log_info(ii).mrk.pos(problem_ind(jj));
      log_info(kk).segments.ival = log_info(kk).segments.ival+log_info(ii).mrk.pos(problem_ind(jj));
    end
  end
  
end
% check if the log_fb-positions make sense.
for ii = 1:length(log_fb)
  problem_ind = find(diff(log_fb(ii).mrk.pos)<0);
  if length(problem_ind)>0
    % something went wrong when concatenating logfiles.
    fprintf('fixing log_fb(%i)\n',ii);
    for jj = 1:length(problem_ind)
      log_fb(ii).mrk.pos(problem_ind(jj)+1:end) = log_fb(ii).mrk.pos(problem_ind(jj)+1:end)+log_fb(ii).mrk.pos(problem_ind(jj));
    end
    for kk = ii+1:length(log_fb)
      % make the logfiles' positions increasing over all logfiles.
      %log_fb(kk).mrk.pos = log_fb(kk).mrk.pos+log_fb(ii).mrk.pos(problem_ind(jj));
   end
  end
  
end


for i = 1:length(log_info)
  if log_info(i).fs~=log_info(1).fs
    error('the logfiles should have the same sampling frequency');
  end
end


for i = 1:length(log_fb)
  if log_fb(i).fs~=log_info(1).fs
    error('the logfiles should have the same sampling frequency');
  end
end

%% Read segment intervals of the requested file
mm = readMarkerTiming(file,opt.fs);
mmp= cat(1, mm.pos, mm.length);

logf = []; flogf= [];
if isfield(opt,'adaptation')
  if opt.adaptation
    % kick out most of the markers: adaptation always starts with 70.
    Mrk_ind = find(ismember(Mrk.toe,opt.markertypes));
    Mrk = mrk_selectEvents(Mrk,Mrk_ind);
    for ii = 1:length(log_fb)
      log_fb(ii).mrk.indexedByEpochs = {'counter','lognumber'};
      flog_ind = find(ismember(log_fb(ii).mrk.toe,opt.markertypes));
      log_fb(ii).mrk = mrk_selectEvents(log_fb(ii).mrk,flog_ind);
    end
    for ii = 1:length(log_info)
      log_ind = find(ismember(log_info(ii).mrk.toe,opt.markertypes));
      log_info(ii).mrk = mrk_selectEvents(log_info(ii).mrk,log_ind);
    end
  end
end

for seg = 1:length(mm.pos)
  % for all segments of the EEG file: look at the respective 
  % markers (mark)
  if seg == length(mm.pos)
    ind = find(ceil(Mrk.pos/Mrk.fs*log_info(1).fs)>=mm.pos(seg));
  else
    ind = find(ceil(Mrk.pos/Mrk.fs*log_info(1).fs)>=mm.pos(seg) & ceil(Mrk.pos/Mrk.fs*log_info(1).fs)<mm.pos(seg+1));
  end
  mark = mrk_selectEvents(Mrk,ind);
  mark.pos = ceil(mark.pos/mark.fs*log_info(1).fs);
  mark.fs = log_info(1).fs;
  collect = []; tryso = 0;
  
  % if length(mark.pos)<10, it will be impossible to align anything.
  % (just because mark.pos will fit almost everywhere).
  if length(mark.pos)<10
    warning('too few markers in this segment! Skipping.');
    continue
  end
  
  while isempty(collect) & tryso<20 & length(mark.toe)>0
    % for the markers in the EEG segments:
    % try to align the feedback markers accordingly.
    for lo = 1:length(log_info)
      % for each logfile: keep the first marker and try to align.
      in = find(mark.toe(1)==log_info(lo).mrk.toe);
      for k = 1:length(in)
        mr_rel = log_info(lo).mrk.toe(in(k):end);
        if length(mr_rel)>= length(mark.toe)
          mr_rel = mr_rel(1:length(mark.toe));
          if sum(mr_rel(1:min(100,length(mr_rel))) ~= mark.toe(1:min(100,length(mark.toe))))==0
            mr_pos = log_info(lo).mrk.pos(in(k):in(k)+length(mark.toe)-1);
            mr_pos = mr_pos-mr_pos(1)+mark.pos(1);
	    % look if the first few markers match.
            if sum(abs(mr_pos(1:min(100,length(mr_pos)))-mark.pos(1:min(100,length(mr_pos))))>4)==0
              collect = cat(1,collect,[lo,in(k)]);
            end
          end
        end
      end
    end
    if isempty(collect)
      ind = ind(2:end);
      mark = mrk_selectEvents(Mrk,ind);
      mark.pos = ceil(mark.pos/mark.fs*log_info(1).fs);
      mark.fs = log_info(1).fs;
      fprintf('Skip one marker\n');
    end
    tryso = tryso+1;
  end
  
  if size(collect,1)==0 & length(mark.toe)>0
    keyboard
    error('nothing found');
  end
  if size(collect,1)>1
    keyboard
    error('too much found');
  end
  
  if length(mark.toe)>0
    % for the EEG-markers: find feedback and classifier logfiles.
    
    %logi: collection of classifier logfiles
    logi = select_logfile(log_info(collect(1)), ...
                          [mmp(seg) mmp(seg+1)]-mark.pos(1) + ...
                          log_info(collect(1)).mrk.pos(collect(2)), ...
                          -mark.pos(1) + ...
                          log_info(collect(1)).mrk.pos(collect(2)));
    % flogi: collection of feedback logfiles
    flogi = select_feedback(log_fb, ...
                            log_info(collect(1)).continuous_file_number, ...
                            [mmp(seg) mmp(seg+1)]- mark.pos(1) + ...
                            log_info(collect(1)).mrk.pos(collect(2)), ...
                            -mark.pos(1) + ...
                            log_info(collect(1)).mrk.pos(collect(2)));
    
    if ~isempty(flogi.mrk.toe)
      if isfield(opt,'adaptation')
	if opt.adaptation
	  % an initial marker 70 should appear.
	  flogi.mrk.indexedByEpochs = {'counter','lognumber'};
	  flogi_ind = min(find(flogi.mrk.toe==70)):length(flogi.mrk.toe);
	  flogi.mrk = mrk_selectEvents(flogi.mrk,flogi_ind);
	  logi_ind = min(find(logi.mrk.toe==70)):length(logi.mrk.toe);
	  logi.mrk = mrk_selectEvents(logi.mrk,logi_ind);
	  Mrk_ind = min(find(Mrk.toe==70)):length(Mrk.toe);
	  Mrk = mrk_selectEvents(Mrk,Mrk_ind);
	end
      else
	%Guidos method.
	ind = find(logi.mrk.pos<ceil(Mrk.pos(1)/Mrk.fs*log_info(1).fs));
	logi.mrk.pos(ind) = [];
	logi.mrk.toe(ind) = [];
	ind = find(flogi.mrk.pos<ceil(Mrk.pos(1)/Mrk.fs*log_info(1).fs));
	flogi.mrk.pos(ind) = [];
	flogi.mrk.toe(ind) = [];
	flogi.mrk.counter(ind) = [];
	flogi.mrk.lognumber(ind) = [];
	pp = find(logi.mrk.pos>=flogi.mrk.pos(1));
	logi.mrk.pos = logi.mrk.pos(pp);
	logi.mrk.toe = logi.mrk.toe(pp);
	del = 100;
	nl = min(find(diff(logi.mrk.pos)>del));
	
	while any(diff(logi.mrk.pos(1:nl))>del-20)
	  del = del+10;
	  nl = min(find(diff(logi.mrk.pos)>del));
	end
	nf = min(find(diff(flogi.mrk.pos)>del & flogi.mrk.pos(2:end)+40>=logi.mrk.pos(nl+1)));
	
	if nl>nf
	  logi.mrk.pos(1:nl-nf) = [];
	  logi.mrk.toe(1:nl-nf) = [];
	elseif nl<nf
	  flogi.mrk.pos(1:nf-nl) = [];
	  flogi.mrk.toe(1:nf-nl) = [];
	  flogi.mrk.counter(1:nf-nl) = [];
	  flogi.mrk.lognumber(1:nf-nl) = [];
	end
      end
      
      % repair fields
      % calculate offset to Mrk
      kk = min(find(ceil(Mrk.pos/Mrk.fs*log_info(1).fs)>=logi.mrk.pos(1)));
      poi =[kk,1,1]; 
      delay = 6;
      while poi(1)<=length(Mrk.toe) |  poi(2)<=length(logi.mrk.toe) | poi(3)<=length(flogi.mrk.toe)
        fprintf('\r %d,%d,%d      ',poi);
        if poi(1)>length(Mrk.toe)
          if poi(2)>length(logi.mrk.toe)
	    %            Mrk.toe = [Mrk.toe,flogi.mrk.toe(poi(3))];
	    %            Mrk.pos = [Mrk.pos,(flogi.mrk.pos(poi(3))-delay)*Mrk.fs/log_info(1).fs];
	    keyboard
	    flogi.mrk.toe(poi(3):end) = [];
	    flogi.mrk.pos(poi(3):end) = [];
	    flogi.mrk.lognumber(poi(3):end) = [];
	    flogi.mrk.counter(poi(3):end) = [];
	    continue;
          else
            Mrk.toe = [Mrk.toe,logi.mrk.toe(poi(2))];
            Mrk.pos = [Mrk.pos,logi.mrk.pos(poi(2))*Mrk.fs/log_info(1).fs];
          end
        end
        if poi(2)>length(logi.mrk.toe)
          logi.mrk.toe = [logi.mrk.toe,Mrk.toe(poi(1))];
          logi.mrk.pos = [logi.mrk.pos,ceil(Mrk.pos(poi(1))/Mrk.fs*log_info(1).fs)];
        end
        if poi(3)>length(flogi.mrk.toe)
          flogi.mrk.toe = [flogi.mrk.toe,Mrk.toe(poi(1))];
          flogi.mrk.pos = [flogi.mrk.pos,ceil(Mrk.pos(poi(1))/Mrk.fs*log_info(1).fs)+delay];
          flogi.mrk.counter = [flogi.mrk.counter,flogi.mrk.counter(end)+1];
          flogi.mrk.lognumber = [flogi.mrk.lognumber,flogi.mrk.lognumber(end)];
        end
        
        if logi.mrk.toe(poi(2))~=Mrk.toe(poi(1)) |  logi.mrk.pos(poi(2))~=ceil(Mrk.pos(poi(1))/Mrk.fs*log_info(1).fs)
          error('');
        end
        
        if flogi.mrk.toe(poi(3))-logi.mrk.toe(poi(2))~=0
	  % current markers are not equal.
	  if isfield(opt,'adaptation')
	    if opt.adaptation
	      % Just look for the temporal alignment.
	      fprintf('Markers don''t match!');
	      if diff(flogi.mrk.pos(poi(3)-1:poi(3)))<...
		    diff(logi.mrk.pos(poi(2)-1:poi(2)))
		% this means a marker is missing in logi.
		fprintf('Inserted one marker into logi.\n');
		logi.mrk.toe = [logi.mrk.toe(1:poi(2)-1), flogi.mrk.toe(poi(3)), logi.mrk.toe(poi(2):end)];
		logi.mrk.pos = [logi.mrk.pos(1:poi(2)-1), flogi.mrk.pos(poi(3))-delay, logi.mrk.pos(poi(2):end)];
	      else
		% this means a marker is missing in flogi.
		  fprintf('Inserted one marker into flogi.\n');
		  flogi.mrk.toe = [flogi.mrk.toe(1:poi(3)-1), logi.mrk.toe(poi(2)), flogi.mrk.toe(poi(3):end)];
		  flogi.mrk.pos = [flogi.mrk.pos(1:poi(3)-1), logi.mrk.pos(poi(2))-delay, flogi.mrk.pos(poi(3):end)];
		  flogi.mrk.counter = [flogi.mrk.counter(1:poi(3)-1), flogi.mrk.counter(poi(3)-1)+1, flogi.mrk.counter(poi(3):end)];
		  flogi.mrk.lognumber = [flogi.mrk.lognumber(1:poi(3)-1), flogi.mrk.lognumber(poi(3)-1), flogi.mrk.lognumber(poi(3):end)];
	      end
	      if Mrk.toe(poi(1))~=logi.mrk.toe(poi(2))
		% also insert the marker into Mrk.
		fprintf('Inserted one marker into Mrk.\n');
		Mrk.toe = [Mrk.toe(1:poi(1)-1), logi.mrk.toe(poi(2)), Mrk.toe(poi(1):end)];
		Mrk.pos = [Mrk.pos(1:poi(1)-1), ceil(logi.mrk.pos(poi(2))/log_info(1).fs*Mrk.fs), Mrk.pos(poi(1):end)];
	      end
	      poi = poi+1
	    end
	  else
	    if problem_marker_bit & flogi.mrk.toe(poi(3))-logi.mrk.toe(poi(2))==1
	      % markers differ only by a non-recorded bit of the parport.
	      logi.mrk.toe(poi(2)) = logi.mrk.toe(poi(2))+1;
	      Mrk.toe(poi(1)) = Mrk.toe(poi(1))+1;
	      poi = poi+1;
	      keyboard
	    else
	      
	      typi = 0;
	      if poi(2)<length(logi.mrk.toe) & flogi.mrk.toe(poi(3))-logi.mrk.toe(poi(2)+1)==0 | (problem_marker_bit & flogi.mrk.toe(poi(3))-logi.mrk.toe(poi(2)+1)==1)
		% flogi has skipped a marker
		typi = 1;
	      end
	      if poi(3)<length(flogi.mrk.toe) & flogi.mrk.toe(poi(3)+1)-logi.mrk.toe(poi(2))==0 | (problem_marker_bit & flogi.mrk.toe(poi(3)+1)-logi.mrk.toe(poi(2))==1)
		% logi has skipped a marker
		typi = typi+2;
	      end
	      if poi(2)<length(logi.mrk.toe) & poi(3)<length(flogi.mrk.toe) & flogi.mrk.toe(poi(3)+1)==logi.mrk.toe(poi(2)+1) | (problem_marker_bit & flogi.mrk.toe(poi(3)+1)-logi.mrk.toe(poi(2)+1)==1)
		% both flogi and logi have skipped.
		typi = 0;
	      end
	      if typi==3
		fprintf(' This case does not make sense.\n');
		keyboard
		h = flogi.mrk.toe(poi(3));
		flogi.mrk.toe(poi(3)) = flogi.mrk.toe(poi(3)+1);
		flogi.mrk.toe(poi(3)+1) = h;
		h = flogi.mrk.pos(poi(3));
		flogi.mrk.pos(poi(3)) = flogi.mrk.pos(poi(3)+1);
		flogi.mrk.pos(poi(3)+1) = h;
		h = flogi.mrk.counter(poi(3));
		flogi.mrk.counter(poi(3)) = flogi.mrk.counter(poi(3)+1);
		flogi.mrk.counter(poi(3)+1) = h;
		h = flogi.mrk.lognumber(poi(3));
		flogi.mrk.lognumber(poi(3)) = flogi.mrk.lognumber(poi(3)+1);
		flogi.mrk.lognumber(poi(3)+1) = h;
	      elseif typi==0
		warning('Marker corrected');
		keyboard
		logi.mrk.toe(poi(2)) = flogi.mrk.toe(poi(3));
		Mrk.toe(poi(1)) = flogi.mrk.toe(poi(3));
		poi = poi+1;
	      elseif typi==1
		fprintf('Inserted one marker into flogi.\n');
		flogi.mrk.toe = [flogi.mrk.toe(1:poi(3)-1), logi.mrk.toe(poi(2)), flogi.mrk.toe(poi(3):end)];
		flogi.mrk.pos = [flogi.mrk.pos(1:poi(3)-1), logi.mrk.pos(poi(2))-delay, flogi.mrk.pos(poi(3):end)];
		flogi.mrk.counter = [flogi.mrk.counter(1:poi(3)-1), flogi.mrk.counter(poi(3)-1)+1, flogi.mrk.counter(poi(3):end)];
		flogi.mrk.lognumber = [flogi.mrk.lognumber(1:poi(3)-1), flogi.mrk.lognumber(poi(3)-1), flogi.mrk.lognumber(poi(3):end)];
	      else
		fprintf('Removed one marker from flogi.');
		keyboard
		flogi.mrk.toe(poi(3))= [];
		flogi.mrk.pos(poi(3))= [];
		flogi.mrk.counter(poi(3))= [];
		flogi.mrk.lognumber(poi(3))= [];
	      end
	    end
	  end
	else
	  poi = poi+1;
	end
      end
    end
  else
    %mark.toe is empty.
    logi = [];
    flogi = [];
  end
  if isempty(logf);
    logf = logi;
  else
    logf = combine_logs(logf,logi);
  end
  
  if isempty(flogf)
    flogf = flogi;
  else
    flogf = combine_feedbacks(flogf,flogi);
  end
end

