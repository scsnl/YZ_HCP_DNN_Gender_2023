% __________________________________________________________________
% TS extraction for resting state fmri
% ------------------------------------------------------------------

function rest_ts_extraction(subject_index,ConfigFile)

% Show the system information and write log files
warning('off', 'MATLAB:FINITE:obsoleteFunction')
c     = fix(clock);
disp('==================================================================');
fprintf('TS extraction for rsfmri starts at %d/%02d/%02d %02d:%02d:%02d\n',c);
disp('==================================================================');
disp(['Current directory is: ',pwd]);
disp('------------------------------------------------------------------');

% dependencies (spm and marsbar for time series extraction)
addpath(genpath('TOOLBOX_DIR/spm12'));
addpath(genpath('TOOLBOX_DIR/marsbar-0.44'));
addpath(genpath('SCRIPTS_DIR/brainImaging/mri/fmri/extract_timeseries/spm12/utils'));
currentdir = pwd

ConfigFile = strtrim(ConfigFile);
if ~exist(ConfigFile, 'file')
	error('cannot find the configuration file')
end
configtext=fileread(ConfigFile);
eval(configtext);

clear ConfigFile;

disp('-------------- Contents of the Parameter List --------------------');
disp(paralist);
disp('------------------------------------------------------------------');

% -------------------------------------------------------------------------
% Read in parameters
% -------------------------------------------------------------------------

subject_i	     	= subject_index;
subject_list        = strtrim(paralist.subjectlist);
runlist             = strtrim(paralist.runlist);
project_dir         = strtrim(paralist.projectdir);
preproc_dir    		= strtrim(paralist.preprocessed_dir);
pipeline		    = strtrim(paralist.pipeline);
roi_dir			    = strtrim(paralist.roi_dir);
roi_list 			= strtrim(paralist.roi_list);
output_dir		    = strtrim(paralist.output_dir);
TR                  = double(paralist.TR);
filter_data         = strtrim(paralist.filter_data);

% init variables
%subjtxtfile = fopen(subjectlist);
%subjectlist=textscan(subjtxtfile,'%s %s %s','Delimiter',',','HeaderLines',1);
%fclose(subjtxtfile);
%subject           = subjectlist(subject_i);
%subject           = char(pad(string(subject),4,'left','0'));
%visit             = num2str(subjectlist(subject_i,2));
%session           = num2str(subjectlist(subject_i,3));

%numsub				= length(subjectlist);
roilist = ReadList(roi_list);
num_roi = length(roilist);

%output_dir = fullfile(project_dir,'results','taskfmri','connectivity','bsds');
%mkdir(output_dir);
mkdir(output_dir);

cd(output_dir);
% extract time series
disp(paralist)
if isfield(paralist,'roi_type')
	if paralist.roi_type == 'atlas'
	    if isfield(paralist,'sumfunc')
		if strcmp(paralist.sumfunc,'data_aug')
				data_aug_perc=paralist.data_aug_perc;
				num_datasets = paralist.data_aug_num_datasets;
				data = extract_roi_ts_rest_subject_atlas_data_aug(project_dir,subject_list,subject_index,runlist,roi_list,roi_dir,preproc_dir,pipeline,TR,output_dir,filter_data,paralist.sumfunc,data_aug_perc,num_datasets);
			else
				if isfield(paralist,'physio_method')
				data = extract_roi_ts_rest_subject_atlas_acompcor(project_dir,subject_list,subject_index,runlist,roi_list,roi_dir,preproc_dir,pipeline,TR,output_dir,filter_data,paralist.sumfunc);
					else
		    		data = extract_roi_ts_rest_subject_atlas(project_dir,subject_list,subject_index,runlist,roi_list,roi_dir,preproc_dir,pipeline,TR,output_dir,filter_data,paralist.sumfunc);
				end
			end
		else
		     if isfield(paralist,'physio_method')
                                data = extract_roi_ts_rest_subject_atlas_acompcor(project_dir,subject_list,subject_index,runlist,roi_list,roi_dir,preproc_dir,pipeline,TR,output_dir,filter_data,paralist.sumfunc);
                     else
                                data = extract_roi_ts_rest_subject_atlas(project_dir,subject_list,subject_index,runlist,roi_list,roi_dir,preproc_dir,pipeline,TR,output_dir,filter_data,paralist.sumfunc);
                     end
		end
	end
else
	data = extract_roi_ts_rest_subject(project_dir,subject_list,subject_index,runlist,roi_list,roi_dir,preproc_dir,pipeline,TR,output_dir,filter_data);
end
cd(output_dir);
fprintf('TS extraction finished.\n');


%% prologue
cd(currentdir);

c     = fix(clock);
disp('==================================================================');
fprintf('TS extraction for rsfmri finishes at %d/%02d/%02d %02d:%02d:%02d\n',c);
disp('==================================================================');
clear all;
