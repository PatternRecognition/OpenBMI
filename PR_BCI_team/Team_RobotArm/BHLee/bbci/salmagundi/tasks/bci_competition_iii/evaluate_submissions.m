compet_dir= [DATA_DIR 'eegImport/bci_competition_iii/berlin/'];
sub_dir= [DATA_DIR 'results/bci_competition_iii_submissions/'];

%data_sets= {'IVa', 'IVb', 'IVc'};
data_sets= {'IVa', 'IVc'};

for di= 1:length(data_sets),
  dd= dir([sub_dir data_sets{di} '_*']);
  nSub(di)= length(dd);
  for si= 1:length(dd),
    ssub_dir= dd(si).name;
    iu= find(ssub_dir=='_');
    name= strrep(ssub_dir(iu(1)+1:end), '_', ' ');
    name(1)= upper(name(1));
    name(diff(iu)+1)= upper(name(diff(iu)+1));
    competitor{si,di}= name;
    switch(data_sets{di}),
     case 'IVa',
      subjects= {'aa','al','av','aw','ay'};
      for vi= 1:length(subjects),
        subj= subjects{vi};
        ddd= dir([sub_dir ssub_dir '/*result*' subj '*.txt']);
        if length(ddd)~=1,
          error('haeh?');
        end
        res_file= [sub_dir ssub_dir '/' ddd.name];
        est_y= load(res_file);
        if strcmp(name, 'Yijun Wang'),
          est_y= est_y(:,2);
        end
        if strcmp(name, 'Cedric Simon'),
          est_y= est_y(:);
          est_y= round( -est_y/2 + 1.5 );
        end
        S= load([compet_dir 'data_set_' data_sets{di} '_' subj '_truth']);
        true_y= S.true_y';
        test_idx= S.test_idx;
        train_idx= setdiff(1:length(true_y), test_idx);
        if any(true_y(train_idx)~=est_y(train_idx)),
          error(sprintf('%s (%s): error in training labels', ssub_dir, subj));
        end
        if any(~ismember(est_y, [1 2])),
          error(sprintf('%s (%s): unexpected value', ssub_dir, subj));
        end
        nTest(vi)= length(test_idx);
        hit_IVa(si,vi)= sum(true_y(test_idx)~=est_y(test_idx));
        err_IVa(si,vi)= 100*mean(true_y(test_idx)~=est_y(test_idx));
        if vi==1,
          Est_y= est_y;
          True_y= true_y;
        else
          Est_y(:,vi)= est_y;
          True_y(:,vi)= true_y;
        end
      end
     case 'IVb',
      warning('not implemented yet');
     case 'IVc',
      subj= 'al';
      ddd= dir([sub_dir ssub_dir '/*result*.txt']);
      if length(ddd)~=1,
        error('haeh?');
      end
      res_file= [sub_dir ssub_dir '/' ddd.name];
      est_y= load(res_file);
      if si==1,
        Est_y= est_y;
      else
        Est_y(:,si)= est_y;
      end
      S= load([compet_dir 'data_set_' data_sets{di} '_' subj '_test_truth']);
      true_y= S.true_y';
      err_IVc(si)= mean((true_y-est_y).^2);
     otherwise,
      error('unknown data set');
    end
  end
end

imagesc(err_IVa); colorbar; 
set(gca, 'YTick',1:nSub(1), 'YTickLabel',competitor(:,1));

clf; set(gca, 'ColorOrder',cmap_rainbow(13)); hold on; plot(err_IVa'); hold off
legend(competitor(:,1), -1);

N= length(err_IVc); 
plot(err_IVc, 1:N); 
set(gca, 'YTick',1:N, 'YTickLabel', competitor(:,3), 'YDir','reverse');

ci= 2;
idx= 1:length(true_y);
%[so idx]= sort(true_y);
hp= plot([true_y(idx) Est_y(idx,ci)], '.');
set(hp(1), 'MarkerSize',16, 'Color','y');

[so,i1]= sort(Est_y(:,ci));
[so,i2]= sort(true_y(i1));
idx= i1(i2);
hp= plot([true_y(idx) Est_y(idx,ci)], '.');
set(hp(1), 'MarkerSize',16, 'Color','y');

idx= find(true_y==0);
hist(Est_y(idx, ci));



return


file= [res_dir 'berlin_IVa_results.txt'];
fid= fopen(file, 'wt');
for ii= 1:size(hit_IVa,1),
  fprintf(fid, '%.2f | %.1f | %.1f | %.1f | %.1f | %.1f | %s | |\n', ...
          100*sum(hit_IVa(ii,:))/sum(nTest), ...
          100*hit_IVa(ii,:)./nTest, competitor{ii,1});
end
fclose(fid);



file= [res_dir 'berlin_IVc_results.txt'];
fid= fopen(file, 'wt');
for ii= 1:size(err_IVc,2),
  fprintf(fid, '%.2f | %s | |\n', err_IVc(ii), competitor{ii,2});
end
fclose(fid);
