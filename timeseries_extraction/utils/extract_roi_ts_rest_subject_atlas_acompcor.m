function data = extract_roi_ts_rest_subject_atlas_acompcor(project_dir,subject_list,subject_i,runlist,roi_list,roi_dir,preproc_dir,pipeline,TR,output_dir,filter_data,sumfunc)

% function extract_roi_ts()
%
% description:
%	this function does the following:
%		extracts the time series
%		detrends
%		remove confounds
%		low and high pass filter (band pass filter)
% 		standardizes
%
%	removal of confounds is done orthogonally to the temporal filters per Lindquist, M., Geuter, S., Wager, T., & Caffo, B. (2018). Modular preprocessing pipelines can reintroduce artifacts into fMRI data. bioRxiv, 407676.
%
% dependencies: marsbar,spm
%
% params:
%	- datadir
%
% Carlo de los Angeles, 2019/05/02
%

% Do not change below:
format longG
tic
roilist = ReadList(roi_list);
num_roi = length(roilist);

runs=ReadList(runlist);
numrun 				= length(runs);

rois=roilist;

subj_data=[];
subj_idx=subject_i;

subjtxtfile = fopen(subject_list);
disp(subject_list);
subject_list=textscan(subjtxtfile,'%s %s %s','Delimiter',',','HeaderLines',1);
fclose(subjtxtfile);

subj         = subject_list{1}{subject_i};
subj           = char(pad(string(subj),4,'left','0'));
visit             = subject_list{2}{subject_i};
session           = subject_list{3}{subject_i};


timeseries=[];


toc
for irun=1:numrun
%try
data_dir=fullfile(project_dir,'data','imaging','participants',char(subj),['visit' visit],['session' session],'fmri',char(runs(irun)),preproc_dir);
if exist(fullfile(output_dir,sprintf('%s_visit%s_session%s_%s_%s_%s_ts.mat',char(subj),visit,session,char(runs{irun}),pipeline,sumfunc)));
   		continue
	elseif ~exist(fullfile(data_dir,[pipeline,'I.nii.gz']));
		continue
   	end
	 % dependencies (spm and marsbar for time series extraction)
addpath(genpath('TOOLBOX_DIR/spm12'));
addpath(genpath('TOOLBOX_DIR/marsbar-0.44'));
addpath(genpath('SCRIPTS_DIR/brainImaging/mri/fmri/extract_timeseries/spm12/utils'));
	   roi_files = spm_select('FPList', roi_dir, ['.*\.mat$']);
	   rois = maroi(cellstr(roi_files));

	disp(sprintf('extracting time series for %s for run %s using physio method acompcor',subj, char(runs(irun))));
	data_dir=fullfile(project_dir,'data','imaging','participants',char(subj),['visit' visit],['session' session],'fmri',char(runs(irun)),preproc_dir);
	disp(sprintf('MR data is at %s', data_dir));
	tmp_dir = fullfile('/scratch/users',getenv('LOGNAME'), 'tmp_files');

    if ~exist(tmp_dir, 'dir')
      	mkdir(tmp_dir);
    end

	temp_dir=fullfile(tmp_dir,[subj,['visit',visit],['session',session],'_',tempname]);

	if ~exist(temp_dir, 'dir')
        mkdir(temp_dir);
    else
        unix(sprintf('rm -rf %s', temp_dir));
        mkdir(temp_dir);
    end

	if ~exist(fullfile(data_dir,[pipeline,'I.nii']))
	unix(sprintf('cp -aLf %s %s', fullfile(data_dir,[pipeline,'I.nii.gz']), ...
        temp_dir));

	try
		gunzip(fullfile(temp_dir,[pipeline,'I.nii.gz']))
	end
	else
		 unix(sprintf('cp -aLf %s %s', fullfile(data_dir,[pipeline,'I.nii']), ...
        temp_dir));
	end
	% nifti_file = spm_select('ExtFPList', temp_dir, ['^',pipeline,'I.*\.nii']);
    % V       = spm_vol(deblank(nifti_file(1,:)));
    % nframes = V.private.dat.dim(4);
    files = spm_select('ExtFPList', temp_dir, ['^',pipeline,'I.*\.nii'],1:9999);

    run_timeseries=[];

    roi_data_obj = get_marsy(rois{:}, files,sumfunc,'q');
	raw_roi_ts = summary_data(roi_data_obj);
	if filter_data == '1';
		roi_ts_detrended = detrend(raw_roi_ts);
		confounds=load(char(fullfile(project_dir,'data','imaging','participants',char(subj),['visit' visit],['session' session],'fmri',char(runs(irun)),preproc_dir,'rp_I.txt')));
		components=dlmread(char(fullfile(project_dir,'data','imaging','participants',char(subj),['visit' visit],['session' session],'fmri',char(runs(irun)),preproc_dir,'components_file.txt')),'',1);
		confounds=horzcat(confounds,components);
		[~,roi_ts_regressed] = regress_fast(confounds,roi_ts_detrended);
		% try
		% 	[~,~,roi_ts_regressed,~,~] = mvregress(confounds,roi_ts_detrended);
		% catch
		% 	roi_ts_regressed=zeros(size(raw_roi_ts));
		% 	for roi_idx=1:size(raw_roi_ts,2)
		% 		[~,~,roi_ts_regressed(:,roi_idx)]=regress(roi_ts_detrended(:,roi_idx),confounds);
		% 	end
		% end
		roi_ts_filtered=bandpass_final_SPM_ts(TR,.008,.09,roi_ts_regressed);
		roi_ts=roi_ts_filtered;
		roi_ts_norm=normalize(roi_ts);
		%roi_ts_standard=standardize(roi_ts);
	end

	save(fullfile(output_dir,sprintf('%s_visit%s_session%s_%s_%s_%s_ts.mat',char(subj),visit,session,char(runs{irun}),pipeline,sumfunc)),'roi_ts','raw_roi_ts','roi_ts_detrended','roi_ts_regressed','roi_ts_filtered','roi_ts_norm');
% catch
% 	continue
%end
end
toc
data=timeseries;
