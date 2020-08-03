eeg_files;
tex_dir= 'studies/season1/';

nSubjects= length(file);
bpm= cell(3,1);
duration= cell(3,1);
hits= cell(3,1);
misses= cell(3,1);
subject= cell(1, nSubjects);
nofb= [];

for si= 1:nSubjects,
  usc= min(find(file{si}=='_'));
  subject{si}= file{si}(1:usc-1);
  if isempty(feedbacks{si}),
    %% no feedback sessions recorded
    nofb= [nofb, si];
    continue;
  end
  for fbi= 1:3,
    if fbi<3,
      typ= '2d';
    else
      typ= 'basket';
    end
    log_idx= feedbacks{si}{2,fbi};
    for li= 1:length(log_idx),
      file_fb_opt= sprintf('%s%s/log/feedback_%s_fb_opt_%d', EEG_RAW_DIR, ...
                           file{si}, typ, log_idx(li));
      load(file_fb_opt);
      file_fb= sprintf('%s%s/log/feedback_%s_%d.log', EEG_RAW_DIR, ...
                       file{si}, typ, log_idx(li));
      cmd= sprintf('grep bit %s', file_fb);
      [s,res]= unix(cmd);
      if s~=0,
        error('trouble with log file');
      end
      if fbi<3,
        [cc,bm,mi,se]= ...
            strread(res, 'counter = %d %*[^'']''%.1f %*[^ ] (%d''%d)%*[^\n]');
        bpm{fbi}(li,si)= bm;
        duration{fbi}(li,si)= se + 60*mi;
        cmd= sprintf('grep HIT %s', file_fb);
        [s,res]= unix(cmd);
        if s~=0,
          error('trouble with log file');
        end
        %% search last occurence of 'HIT' before bit rate
        res= strread(res, '%s', 'delimiter','\n');
        hc= inf;
        lidx= length(res)+1;
        while hc>cc,
          lidx= lidx-1;
          hc= strread(res{lidx}, 'counter = %d %*[^\n]');
        end
        hi= strread(res{lidx}, '%*[^:]: %d%*[^\n]');
        hits{fbi}(li,si)= hi;
        misses{fbi}(li,si)= fb_opt.matchpoints - hi;
      else
        [cc,bm]= ...
            strread(res, 'counter = %d %*[^'']''%.1f %*[^\n]');
        bpm{fbi}(li,si)= bm;       
        duration{fbi}(li,si)= fb_opt.matchpoints * ...
            (fb_opt.trial_duration + ...
             fb_opt.time_after_hit + ...
             fb_opt.time_before_next + ...
             fb_opt.time_before_free) / 1000;
        cmd= sprintf('grep "number = 6" %s', file_fb);
        [s,res]= unix(cmd);
        if s~=0,
          error('trouble with log file');
        end
        %% search last occurence of 'HIT' before bit rate
        res= strread(res, '%s', 'delimiter','\n');
        hc= inf;
        lidx= length(res)+1;
        while hc>cc,
          lidx= lidx-1;
          hc= strread(res{lidx}, 'counter = %d %*[^\n]');
        end
        [hi,mi]= strread(res{lidx}, '%*[^'']''%d:%d%*[^\n]');
        hits{fbi}(li,si)= hi;
        misses{fbi}(li,si)= mi;
      end
    end
  end
end

subject(nofb)= [];
warning('nTargets must be read from file');
for fbi= 1:3,
  bpm{fbi}(:,nofb)= [];
  duration{fbi}(:,nofb)= [];
  hits{fbi}(:,nofb)= [];
  misses{fbi}(:,nofb)= [];
  
  oki= find(~isnan(hits{fbi}));
  sum_hits{fbi}= sum(hits{fbi});
  sum_misses{fbi}= sum(misses{fbi});
  sum_duration{fbi}= sum(duration{fbi});
  nt= 2 + (fbi==3);
  overall_bpm{fbi}= ...
      bitrate(sum_hits{fbi}./(sum_hits{fbi}+sum_misses{fbi}), nt) .* ...
      (sum_hits{fbi}+sum_misses{fbi})*60./sum_duration{fbi};
  overall{fbi}= [sum_hits{fbi}; sum_misses{fbi}; sum_duration{fbi}; ...
                 overall_bpm{fbi}];
  overall{fbi}= overall{fbi}(:)';

  not_performed= find(duration{fbi}==0);
  bpm{fbi}(not_performed)= NaN;
  duration{fbi}(not_performed)= NaN;
  hits{fbi}(not_performed)= NaN;
  misses{fbi}(not_performed)= NaN;
end

%return 

fbSubjects= length(subject);
fmt_entry= {'%d','%d','%.0f','%.1f'};
fmt_row= repmat(fmt_entry, [1 fbSubjects]);
fmt_table= ['@{}l*{' int2str(fbSubjects) '}{r@{:}l@{\hspace{1.5ex}}r' ...
            '@{$\,$s\hspace{1.5ex}}>{\bfseries}r<{\mdseries}}' ...
            '@{}'];
fb_title= {'1d abs', '1d rel', 'basket'};
for fbi= 1:3,
  maxSessions= size(bpm{fbi},1);
  tab= zeros(maxSessions, fbSubjects*4);
  tab(:,1:4:end)= hits{fbi};
  tab(:,2:4:end)= misses{fbi};
  tab(:,3:4:end)= round(duration{fbi});
  tab(:,4:4:end)= bpm{fbi};
  opt= struct('title', fb_title{fbi}, ...
              'col_title', {subject}, ...
              'fmt_table', fmt_table, ...
              'fmt_row', {fmt_row}, ...
              'row_summary', {{'overall',overall{fbi}}});
  latex_table([tex_dir 'results_' feedbacks{1}{1,fbi}], tab, opt);
end
