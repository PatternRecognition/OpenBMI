compet_dir= [DATA_DIR 'bci_competition_iv/'];

dd= dir([compet_dir 'submissions']);
dd(strmatch('.', {dd.name}))= [];
compet= {dd.name};
nSub= length(compet);
for ii= 1:nSub,
  is= find(isstrprop(compet{ii},'upper'), 2, 'first');
  compet_name{ii}= [compet{ii}(1:is(2)-1) ' ' compet{ii}(is(2):end)];
end
T_crop= 1759140;

clear err*
sbj_code= 'abfgcde';
for si= 1:length(sbj_code),
  sbj= sbj_code(si);
  fprintf('Subject %s\n', sbj);
  
  %% get true labels
  load([compet_dir 'BCICIV_eval_ds1' sbj '_1000Hz_true_y'], 'true_y');
  if si==1,
    true_y= true_y(1:T_crop);
  end
  tt= floor(length(true_y)/10);
  true_y100= mean(reshape(true_y(1:10*tt), [10 tt]), 1)';
  T100= length(true_y100);
  
  %% read submitted labels from competitors
  for n= 1:nSub,
    tmp_y= load([compet_dir 'submissions/' dd(n).name ...
                 '/Result_BCIC_IV_ds1' sbj '.txt']);
    if strcmp(compet(n),'FabianBachl'),
      tmp_y= tmp_y(1:2:end);
    end
    if strcmp(compet(n),'SungWook'),
      tmp_y= tmp_y(3,:)';
    end
    fprintf('  %7d samples submitted by %s.\n', length(tmp_y), compet{n});
    if length(tmp_y)> 300000,
      est_y= zeros(length(true_y), 1);
      T0= min(length(true_y), length(tmp_y));
      est_y(1:T0)= tmp_y(1:T0);
      err(si,n)= nanmean((true_y-est_y).^2);
    else
      est_y= zeros(T100, 1);
      T0= min(T100, length(tmp_y));
      est_y(1:T0)= tmp_y(1:T0);
      err(si,n)= nanmean((true_y100-est_y).^2);
    end
%    idx= find(true_y100==-1);
%    err1(si,n)= nanmean((true_y100(idx)-est_y(idx,n)).^2);
%    idx= find(true_y100==0);
%    err2(si,n)= nanmean((true_y100(idx)-est_y(idx,n)).^2);
%    idx= find(true_y100==1);
%    err3(si,n)= nanmean((true_y100(idx)-est_y(idx,n)).^2);
  end
end

err_avg= mean(err(1:4,:));
[so,si]= sort(err_avg);

clf;
axes('Position',[0.2 0.1 0.75 0.85]);
imagesc(err(:,si)'); colorbar; 
set(gca, 'YTick',1:nSub, ...
         'YTickLabel', strcat(compet_name(si), cprintf('  (%.2f)', ...
                                                  err_avg(si))'));
%         'XTickLabel','a|b|c|d|e|f|g');


save([DATA_DIR 'results/bci_competition_iv/results_ds1'], ...
     'err', 'compet_name');

info_file=  [DATA_DIR 'bci_competition_iv/evaluation/further_info_ds1.txt'];
[allnames, lab, method]= textread(info_file, '%s%s%s\n', 'delimiter','|');
name= allnames;
coname= cell(1, length(name));
for ii= 1:length(name),
  is= min(find(name{ii}==','));
  if ~isempty(is),
    coname{ii}= strtrim(name{ii}(is+1:end));
    name{ii}= strtrim(name{ii}(1:is-1));
  else
    name{ii}= strtrim(name{ii});
  end
end

res_dir= [DATA_DIR 'bci_competition_iv/evaluation/'];
file= [res_dir 'evaluation_ds1.txt'];
fid= fopen(file, 'wt');
for ii= 1:size(err,2),
  k= si(ii);
  ij= strmatch(compet_name{k}, name);
  fprintf(fid, '%.3f | %.3f | %.3f | %.3f | %.3f | %s | %s | %s\n', ...
          mean(err(1:4,k)), err(1:4,k), allnames{ij}, lab{ij}, method{ij});
end
fclose(fid);


res_dir= [DATA_DIR 'bci_competition_iv/evaluation/'];
file= [res_dir 'evaluation_ds1a.txt'];
fid= fopen(file, 'wt');
for ii= 1:size(err,2),
  k= si(ii);
  ij= strmatch(compet_name{k}, name);
  fprintf(fid, '%.3f | %.3f | %.3f | %.3f | %s | %s | %s\n', ...
          mean(err(5:7,k)), err(5:7,k), allnames{ij}, lab{ij}, method{ij});
end
fclose(fid);




return;

clf; set(gca, 'ColorOrder',cmap_rainbow(nSub)); hold on; plot(err); hold off
legend(compet, -1);

plot([mean(err1(:,si))' mean(err2(:,si))' mean(err3(:,si))'])
errMI= mean(cat(3,err1,err3),3);
plot([mean(errMI(:,si))' mean(err2(:,si))']);


return
