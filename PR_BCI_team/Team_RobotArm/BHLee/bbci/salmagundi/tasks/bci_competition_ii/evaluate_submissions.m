name_list= {'zhang', 'neal', 'torre', 'saffari', 'chung', 'huang', ...
            'roerig','brugger', 'schroeder', 'rissacher', 'smith', ...
            'hoffmann', 'delriovera', 'mensh', 'mbwana'};

sub_dir= [EEG_IMPORT_DIR 'bci_competition_ii_submissions/'];

cd([BCI_DIR 'tasks/bci_competition_ii']);
load('our_export');
y_truth= [0 1]*mrk.y(:,test_idx);
%fid= fopen('labels_data_set_iv.txt', 'w');
%fprintf(fid, '%d\n', round(y_truth));
%fclose(fid);

nSubmissions= length(name_list);
for ii= 1:nSubmissions,
  file= [sub_dir name_list{ii} '_result'];
  if exist([file '.mat'], 'file'),
    load([file '.mat']);
  else
    y_test= load([file '.txt']);
  end
  if any(y_test<0),
    y_test(find(y_test<0))= 0;
  end
  err(ii)= sum(y_test(:)~=y_truth(:));
  fprintf('%12s:  ', name_list{ii});
  uu= unique(y_test);
  for jj= 1:length(uu),
    fprintf('%d [%d]  ', uu(jj), sum(y_test==uu(jj)));
  end
  fprintf('\n');
end
fprintf('\n');

[so,si]= sort(err);
for uu= 1:nSubmissions,
  ii= si(uu);
  rk= 1+sum(err<err(ii));
  fprintf('%2d> %12s: %g%%\n', rk, name_list{ii}, err(ii));
end

barh(err(si))
set(gca, 'yDir','reverse', 'yTickLabel',name_list(si))
xlabel('error on competition test set [%]');
