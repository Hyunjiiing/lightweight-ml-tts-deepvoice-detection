=====================================================================================================
ASVspoof 2019: The 3rd Automatic Speaker Verification Spoofing and Countermeasures Challenge database

Physical access (PA)
=====================================================================================================


1. Directory Structure
_______________________

  --> PA 
          --> ASVspoof2019_PA_asv_protocols
          --> ASVspoof2019_PA_asv_scores
	  --> ASVspoof2019_PA_cm_protocols
          --> ASVspoof2019_PA_dev
          --> ASVspoof2019_PA_eval
	  --> ASVspoof2019_PA_train
	  --> README.PA.txt


2. Description of the audio files
_________________________________

   ASVspoof2019_PA_train, ASVspoof2019_PA_dev, and ASVspoof2019_PA_eval contain audio files for training, development, and evaluation
   (PA_T_*.flac, PA_D_*.flac, and PA_E_*.flac, respectively). ASVspoof2019_PA_dev, and ASVspoof2019_PA_eval contain audio files to enroll ASV system. The audio files in the directories are in the flac format. 
   The sampling rate is 16 kHz, and stored in 16-bit.
    
    
3. Description of the protocols
_______________________________

CM protocols:

   ASVspoof2019_PA_cm_protocols contains protocol files in ASCII format for ASVspoof countermeasures:

   ASVspoof2019.PA.cm.train.trn.txt: training file list
   ASVspoof2019.PA.cm.dev.trl.txt: development trials
   ASVspoof2019.PA.cm.eval.trl.txt: evaluation trials

   Each column of the protocol is formatted as:
   
   SPEAKER_ID AUDIO_FILE_NAME ENVIRONMENT_ID ATTACK_ID KEY

   	1) SPEAKER_ID: 		PA_****, a 4-digit speaker ID
   	2) AUDIO_FILE_NAME:     name of the audio file
   	3) ENVIRONMENT_ID:	a triplet (S,R,D_s), which take one letter in the set {a,b,c} as categorical value, defined as:

								a		b		c
			--------------------------------------------------------------------------------
			S:   Room size (square meters)		2-5		5-10		10-20
			R:   T60 (ms)				50-200		200-600		600-1000
			D_s: Talker-to-ASV distance (cm)	10-50		50-100		100-150


   	4) ATTACK_ID:		a duple (D_a,Q), which take one letter in the set {A,B,C} as categorical value, defined as

								A		B		C
			-----------------------------------------------------------------------------
			Z: Attacker-to-talker distance (cm)	10-50		50-100		> 100
			Q: Replay device quality		perfect		high		low

			for bonafide speech, ATTACK_ID is left blank ('-')
		
   	5) KEY:			'bonafide' for genuine speech, or, 'spoof' for spoofing speech


   Definition of Replay Device Quality (Q)
   _______________________________________


			Replay device quality
			=====================

					OB (kHZ)	minF (Hz)	linearity (dB)
			--------------------------------------------------------------
			Perfect		inf		0		inf		
			High		> 10		< 600		> 100
			Low		< 10		> 600		< 100

			where:	"OB" is the occupied bandwidth
				"minF" is the lower bound of OB
				"linearity" is the linear/non-linear OB power difference

ASV protocols:

   ASVspoof2019_PA_asv_protocols contains the protocol files for ASV system

   ASVspoof2019.PA.asv.<1>.<2>.<3>.txt
	where
	<1> is either 'dev' or 'eval' based on whether the files describe the development or evaluation protocol,
	<2> ie either 'male (m)' or 'female (f)' separating the genders from each other or 'gender independent (gi)' 
	    contains trials for both genders (male trials followed by female trials),
	<3> is either 'trl' or 'trn' (trl = trial list, trn = speaker enrollment list).

	Trial (trl) file format:
	1st column: claimed speaker ID
	2nd column: test file ID
	3rd column: environment ID
	4rd column: spoof attack ID (or 'bonafide' if the speech is not spoofed)
	5th column: key (target = target trial, nontarget = impostor trial, spoof = spoofing attack)

	Enrollment (trn) file format:
	1st column: ID of enrolled speaker per environment
	2nd column: IDs of files used in the enrollment separated by commas
	
	
4. Baseline ASV scores
______________________

   ASVspoof2019_PA_asv_scores contains the scores calculated by a baseline ASV system for t-DCF evaluation
   
   	ASVspoof2019.PA.asv.dev.gi.trl.scores.txt:  scores given by the ASV system for development set data
   	ASVspoof2019.PA.asv.eval.gi.trl.scores.txt: scores given by the ASV system for evaluation set data	
   	
   Each column is formatted as:
   
   CM_KEY ASV_KEY SCORES

   	1) CM_KEY: 		'bonafide' for genuine speech, or, the ID of the spoofing attack (AA - CC)
   	2) ASV_KEY: 		'target' for claimed speaker, or, 'nontarget' for impostor speaker, or, 'spoof' for spoofing speech
   	3) SCORES: 		similarity score value			

-------------------------------------------------------------------------------------------------------------------------------------

References:

D. R. Campbell, K. J. Palomäki, and G. Brown, "A MATLAB simulation of "shoebox" roomacoustics for use in research and teaching,"
Computing and Information Systems Journal, ISSN1352-9404, vol. 9, no. 3, 2005.

E. Vincent. (2008) Roomsimove. [Online]. Available: http://homepages.loria.fr/evincent/software/Roomsimove_1.4.zip

A. Novak, P. Lotton, and L. Simon, "Synchronized swept-sine: Theory, application, and implementation,"
Journal of the Audio Engineering Society, vol. 63, no. 10, pp. 786–798, 2015.
[Online].  Code available: https://ant-novak.com/pages/sss/

