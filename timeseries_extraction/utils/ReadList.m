
function [outlist] = ReadList(inlist)

inlist = strtrim(inlist);
if iscell(inlist)
  [pathstr, name, ext] = fileparts(inlist{1});
  if strcmp(ext,'.txt')
    inlist_file = inlist{1};
    if ~exist(inlist_file,'file')
      error('Cannot find file: %s \n', inlist_file);
    else
      outlist = GetList(inlist_file);
    end
  else
    outlist = strtrim(inlist);
  end
else
  [pathstr, name, ext] = fileparts(inlist);
  if strcmp(ext,'.txt')
    inlist_file = inlist;
    if ~exist(inlist_file,'file')
      error('Cannot find file: %s \n', inlist_file);
    else
      outlist = GetList(inlist_file);
    end
  else
    outlist = strtrim({inlist});
  end
end

end

function slist = GetList(filename)

fid = fopen(filename);
cnt = 1;

while ~feof(fid)
    fstr = fgetl(fid);
    str  = strtrim(fstr);
    if ~isempty(str)
          slist{cnt} = str;
          cnt = cnt + 1;
    end
end

fclose(fid);

end
