function prepareSelfpacedDir(dirName, filterList, paceList)
%prepareSelfpacedDir(dirName, <filterList, paceList>)

global EEG_RAW_DIR

if ~exist('filterList','var'), filterList={'raw','cut50'}; end
if ~exist('paceList','var'), paceList= [5 2 1 0.5 0.3]; end


iu= find(dirName=='_');
subject= dirName(1:iu-1);
for ip= 1:length(paceList),
  paceName= num2str(paceList(ip));
  paceName(paceName=='.')= '_';
  fileName= fullfile(dirName, ['selfpaced' paceName 's' subject]);
  if exist([EEG_RAW_DIR fileName '.eeg'], 'file'),
    fprintf('processing %s\n', fileName);
    prepareSelfpaced(fileName, filterList, 1000*paceList(ip)/2);
  end
end
