%
% 2010-11-02 Bastian Venthur <bastian.venthur@tu-berlin.de>
%
% Make experiments reproducible by gathering information about the exact files
% which where used.
% Currently we save the following info:
% - svn info (to get the svn revision)
% - svn status (to check for modified or untracked files)
% - svn diff (to see the difference in tracked and modified files)
% - the matlab path
% - we check the matlab path for directories which are not in the standard
%   toolbox or matlab path, we also save the contents of those files
%
% Requirements:
% 
% To use this script on UNIX you have to have the 'svn' package installed.
% To use it on Windows you need win32svn
% <http://sourceforge.net/projects/win32svn/>, it can be installed savely
% besides tortoise.
function main()
    % FIXME: I use git-svn so BCI_DIR points to a git repository
    % BCI_DIR = '/home/venthur/svn/_bbci/';
    global BCI_DIR;
    % TODO: files and dirs we create are also not under revision control, so we
    % have to make sure they don't get copied around every time an experiment
    % starts

    basename = datestr(now, 'yyyy-mm-ddTHH_MM_SS');
    mkdir(basename);
    filename = strcat(basename, '.experiment-info');

    % various svn information
    [status, output] = system(['svn info ' BCI_DIR]);
    write_to_file(output, fullfile(basename, 'svn_info.txt'));

    [status, output] = system(['svn status ' BCI_DIR]);
    write_to_file(output, fullfile(basename, 'svn_status.txt'));

    svnfilelist = check_untracked_svn_files(output);

    [status, output] = system(['svn diff ' BCI_DIR]);
    write_to_file(output, fullfile(basename, 'svn_diff.patch'));

    % output the path
    p = path;
    write_to_file(p, fullfile(basename, 'matlab_path.txt'));

    % strip BCI_DIR and matlabroot from path
    if ~isunix
        % filesep on windows is a special character for regexp, escape it.
        p = path_exclude(p, strrep(matlabroot, '\', '\\'));
        p = path_exclude(p, strrep(BCI_DIR, '\', '\\'));
    else
        p = path_exclude(p, matlabroot);
        p = path_exclude(p, BCI_DIR);
    end
    write_to_file(p, fullfile(basename, 'path_wo_matlabroot_and_bci_dir.txt'));
    
    p = regexp(p, pathsep, 'split');
    p = p';
    pathfilelist = {};
    for i = 1:length(p)
        pathfilelist{length(pathfilelist)+1} = char(p(i));
    end

    % copy files not under revision control
    if ~isempty(svnfilelist)
        svnroot = fullfile(basename, 'svnroot');
        mkdir(svnroot);
        for i = 1:length(svnfilelist)
            fabs = svnfilelist{i};
            frel = fabs(length(BCI_DIR)+1:length(fabs));
            if isdir(fabs)
                mkdir(fullfile(svnroot, frel));
                copyfile(fabs, fullfile(svnroot, frel));
            else
                d = fileparts(frel);
                mkdir(fullfile(svnroot, d));
                copyfile(fabs, fullfile(svnroot, d));
            end
        end
    end

    % copy files not in matlabroot and BCI_DIR
    if ~isempty(pathfilelist)
        root = fullfile(basename, 'fsroot');
        mkdir(root);
        for i = 1:length(pathfilelist)
            dir = pathfilelist{i};
            destdir = dir;
            % on windows destdir will contain a : (d:\foo\bar), which is not
            % allowed in filenames, so we replace it with something else
            if ~isunix
                destdir = strrep(destdir, ':', '_');
            end
            mkdir(fullfile(root, destdir));
            copyfile(dir, fullfile(root, destdir));
        end
    end
end


% Write data to file
function write_to_file(str, filename)
    fh = fopen(filename, 'w');
    fprintf(fh, '%s', str);
    fclose(fh);
end


% Pretty printer
function s = pretty_print(str, header)
    s = sprintf('--- %s ---\n%s\n', header, str);
end


% Check output of svn stat and return absolute path of all files/dirs which are
% not under revision control (i.e. the lines of svn stat which start with a ?)
function files = check_untracked_svn_files(svn_status_output)
    files = {};
    lines = splitlines(svn_status_output);
    for i = 1:length(lines)
        line = char(lines(i));
        if ~isempty(line) && line(1) == '?'
            % svn stat has 7 colums for stuff and the filename starts with the
            % 8th character
            filename = line(8:length(line));
            % apparently on windows, the filename starts with the 9th
            % character
            filename = strtrim(filename);
            files{length(files)+1} = filename;
        end
    end
end


% Split lines by new line characters.
%
% If it finds DOS-style new lines (\r\n) it converts them to UNIX style
% new lines (\n), before splitting.
function s = splitlines(str)
    WIN_NL = '\r\n';
    UNIX_NL = '\n';
    s = strrep(str, WIN_NL, UNIX_NL);
    s = regexp(s, UNIX_NL, 'split');
end

